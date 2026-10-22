"""L0 基线：上下文均值复制（密度帽合规版）。

每个上下文输出 300 扰动 × 400 细胞，每行 = 该上下文 NTC 均值谱（raw counts 空间、
非负整数、csr 稀疏、滤除低表达基因以满足提交密度帽）。

用法:
    python src/models/l0_context_mean/make_predictions.py \
        --config configs/l0_context_mean/default.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[3]
# iter043-53: final 阶段 panel vcc2026-test-1 上下文为 D/E/F (controls_final manifest)。
# 保留 A/B/C 兼容 val 复刻; make_predictions 由 config 的 controls_dir 推断:
# controls_final → D/E/F, 否则 A/B/C。
CONTEXTS_VAL = ("A", "B", "C")
CONTEXTS_FINAL = ("D", "E", "F")
CONTEXTS = CONTEXTS_VAL
# 提交硬约束（virtualcellchallenge.org / vcc-cli wiki, 2026-09 核实）
MAX_TOTAL_ENTRIES = 4_750_000_000
MAX_PER_CELL_COUNTS = 1_000_000
N_PERTS, N_CELLS_PER_PERT = 300, 400


def load_config(path: Path) -> dict:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    # iter043-53: 按 controls_dir 推断上下文集合 (final 阶段 D/E/F)
    global CONTEXTS
    if "controls_final" in str(cfg.get("controls_dir", "")):
        CONTEXTS = CONTEXTS_FINAL
    else:
        CONTEXTS = CONTEXTS_VAL
    return cfg


def context_mean_counts(h5ad_path: Path, min_mean_cpm: float) -> tuple[np.ndarray, dict]:
    """读单上下文 NTC，返回整数化均值谱（基因级）与统计信息。

    min_mean_cpm: 均值低于该 CPM 的基因置零不存储（密度帽 + 去噪）。
    """
    a = ad.read_h5ad(h5ad_path)
    X = a.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr().astype(np.float64)
    mean = np.asarray(X.mean(axis=0)).ravel()  # 每基因平均 UMI

    lib_size = np.asarray(X.sum(axis=1)).ravel()
    median_lib = float(np.median(lib_size))
    # CPM 阈值 -> 绝对 UMI 阈值（按中位库深折算）；下限 1.0：均值四舍五入后 ≥1 count
    min_mean_counts = max(min_mean_cpm * median_lib / 1e3, 1.0)

    keep = mean >= min_mean_counts
    profile = np.where(keep, mean, 0.0)
    profile = np.rint(profile).astype(np.float32)  # raw counts 必须整数值
    stats = {
        "n_cells": int(X.shape[0]),
        "median_lib_size": median_lib,
        "genes_kept": int(keep.sum()),
        "genes_total": int(len(mean)),
        "kept_fraction_of_mass": float(profile.sum() / max(mean.sum(), 1e-9)),
    }
    return profile, stats


def build_submission(cfg: dict) -> tuple[ad.AnnData, dict]:
    controls_dir = ROOT / cfg["controls_dir"]
    gene_names = pd.read_csv(controls_dir / "gene_names.csv")["gene_name"].astype(str)
    perts = pd.read_csv(controls_dir / cfg["pert_counts_file"])["target_gene"].astype(str)
    assert len(gene_names) == 18533 and len(perts) == N_PERTS

    blocks, report = [], {"contexts": {}, "min_mean_cpm": cfg["min_mean_cpm"]}
    for ctx in CONTEXTS:
        t0 = time.time()
        profile, st = context_mean_counts(controls_dir / f"context_{ctx}.h5ad", cfg["min_mean_cpm"])
        st["seconds"] = round(time.time() - t0, 1)
        report["contexts"][ctx] = st

        rows = sp.csr_matrix(np.tile(profile, (N_CELLS_PER_PERT, 1)))
        for g in perts:
            blocks.append(rows)
        print(f"[ctx {ctx}] kept {st['genes_kept']}/{st['genes_total']} genes, "
              f"mass {st['kept_fraction_of_mass']:.3f}, median lib {st['median_lib_size']:.0f}, "
              f"{st['seconds']}s")

    X = sp.vstack(blocks).tocsr().astype(np.float32)
    X.eliminate_zeros()
    obs = pd.DataFrame(
        {
            "target_gene": np.tile(np.repeat(perts.to_numpy(), N_CELLS_PER_PERT), len(CONTEXTS)),
            "context": np.repeat(CONTEXTS, N_PERTS * N_CELLS_PER_PERT),
        },
        index=[f"c{c}_p{p}_cell{i}"
               for c in CONTEXTS
               for p in range(N_PERTS)
               for i in range(N_CELLS_PER_PERT)],
    )
    a = ad.AnnData(X=X, obs=obs)
    a.var_names = gene_names.to_numpy()
    a.var.index.name = None
    return a, report


def validate(a: ad.AnnData, perts: pd.Series) -> dict:
    """提交前静态 schema 校验（密度帽/规模/整数性/每细胞上限）。"""
    assert a.shape == (360_000, 18_533), f"shape {a.shape}"
    assert (a.obs.groupby("context", observed=True)["target_gene"].nunique() == N_PERTS).all()
    assert set(a.obs["target_gene"]) == set(perts), "pert set mismatch"
    assert not (a.obs["target_gene"] == "non-targeting").any(), "NTC rows forbidden"
    assert sp.issparse(a.X) and a.X.dtype == np.float32

    data = a.X.data
    assert (data >= 0).all() and np.all(data == np.rint(data)), "must be non-negative integers"
    per_cell = np.asarray(a.X.sum(axis=1)).ravel()
    max_cell = float(per_cell.max())
    assert max_cell <= MAX_PER_CELL_COUNTS, f"per-cell counts {max_cell:.0f} > 1e6"

    entries = int(a.X.nnz)
    assert entries <= MAX_TOTAL_ENTRIES, f"entries {entries} > cap"
    margin = 1 - entries / MAX_TOTAL_ENTRIES
    # 均值基线本身≈11.8k/细胞，官方余量仅~10%：低于 5% 视为危险
    assert margin >= 0.05, f"density margin {margin:.1%} < 5% (unsafe)"
    return {
        "shape": list(a.shape),
        "entries": entries,
        "entries_per_cell_avg": round(entries / 360_000, 1),
        "density_margin": round(margin, 4),
        "max_per_cell_counts": round(max_cell, 1),
        "integer_valued": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/l0_context_mean/default.yaml"))
    args = ap.parse_args()
    cfg = load_config(Path(args.config))

    out_dir = ROOT / "outputs/submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    a, report = build_submission(cfg)
    report["validation"] = validate(a, pd.read_csv(ROOT / cfg["controls_dir"] / cfg["pert_counts_file"])["target_gene"])

    stem = cfg["output_stem"]
    h5ad_path = out_dir / f"{stem}.h5ad"
    a.write_h5ad(h5ad_path, compression="gzip")
    report["h5ad"] = str(h5ad_path)
    (ROOT / "outputs/metrics" / f"{stem}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["validation"], indent=2))
    print(f"[done] {h5ad_path}")


if __name__ == "__main__":
    main()
