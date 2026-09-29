"""L1 签名迁移基线：公共 CRISPRi 伪批量 logFC → 竞赛上下文。

pred(g, ctx) = round( NTC均值(ctx)_g × 2^(λ·Δ_g) )，Δ=跨数据集聚合签名（18,533 面板映射）。
未覆盖靶基因用 gwps 全局均值签名（通用 CRISPRi 应激响应）小系数兜底。

用法: python src/models/l1_signature/make_predictions.py --config configs/l1_signature/ws-v1.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.models.l0_context_mean.make_predictions import (  # noqa: E402
    CONTEXTS, MAX_TOTAL_ENTRIES, N_CELLS_PER_PERT, N_PERTS, context_mean_counts, load_config,
    validate,
)


def load_delta(cfg: dict, panel: pd.Index, targets: list[str]) -> tuple[pd.DataFrame, dict]:
    """聚合各数据集签名为每靶基因的 Δ 长表 [target_gene, gene, delta]；返回 (长表, 报告)。"""
    sig_dir = ROOT / "data/interim/signatures"
    per_ds = {}
    report = {"datasets": {}}
    for name in cfg["datasets"]:
        f = sig_dir / f"{name}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        df = df[df["gene"].isin(panel)]
        g = df.groupby(["target_gene", "gene"], sort=False)["logfc"].mean().reset_index()
        per_ds[name] = g
        report["datasets"][name] = {"perts": int(df["target_gene"].nunique()),
                                    "genes_in_panel": int(g["gene"].nunique())}

    covered = set().union(*(set(g["target_gene"]) for g in per_ds.values())) if per_ds else set()
    parts = []
    for name, g in per_ds.items():
        part = g[g["target_gene"].isin(targets) & g["target_gene"].isin(covered)].copy()
        # 跨数据集同基因取均值（v1: 等权；后续换签名相关性权重）
        agg = part.groupby(["target_gene", "gene"], sort=False)["logfc"].mean().reset_index()
        parts.append(agg.rename(columns={"logfc": "delta"}))
    real = (pd.concat(parts, ignore_index=True)
            if parts else pd.DataFrame(columns=["target_gene", "gene", "delta"]))
    real = real.groupby(["target_gene", "gene"], sort=False)["delta"].mean().reset_index()

    # 兜底: gwps 全局均值签名（通用应激响应）
    uncovered = [t for t in targets if t not in covered]
    if uncovered and (sig_dir / "gwps.parquet").exists() and cfg["lambda_generic"] > 0:
        gw = pd.read_parquet(sig_dir / "gwps.parquet")
        generic = gw.groupby("gene", sort=False)["logfc"].mean().reset_index()
        generic = generic[generic["gene"].isin(panel)]
        gen = pd.DataFrame({
            "target_gene": np.repeat(uncovered, len(generic)),
            "gene": np.tile(generic["gene"].to_numpy(), len(uncovered)),
            "delta": np.tile(generic["logfc"].to_numpy(), len(uncovered)),
        })
        real = pd.concat([real, gen], ignore_index=True)

    real["delta"] = real["delta"].clip(-cfg["max_abs_delta"], cfg["max_abs_delta"])
    report.update({"covered_targets": len(covered), "uncovered_targets": len(uncovered),
                   "delta_rows": len(real)})
    return real, report


def build_submission(cfg: dict) -> tuple["ad.AnnData", dict]:  # noqa: F821
    import anndata as ad

    controls_dir = ROOT / cfg["controls_dir"]
    gene_names = pd.read_csv(controls_dir / "gene_names.csv")["gene_name"].astype(str)
    panel = pd.Index(gene_names)
    perts = pd.read_csv(controls_dir / cfg["pert_counts_file"])["target_gene"].astype(str).tolist()

    delta, report = load_delta(cfg, panel, perts)
    dmap = delta.set_index(["target_gene", "gene"])["delta"]

    blocks, ctx_stats = [], {}
    sampling = bool(cfg.get("cell_sampling", False))
    rng = np.random.default_rng(cfg.get("seed", 0))
    for ctx in CONTEXTS:
        a_ctx = ad.read_h5ad(controls_dir / f"context_{ctx}.h5ad")
        X_ntc = sp.csr_matrix(a_ctx.X)
        lib = np.asarray(X_ntc.sum(axis=1)).ravel()
        med_lib = float(np.median(lib))
        mean_prof = np.asarray(X_ntc.mean(axis=0)).ravel()
        keep = mean_prof >= max(cfg["min_mean_cpm"] * med_lib / 1e3, 1.0)
        st = {"n_cells": int(X_ntc.shape[0]), "median_lib_size": med_lib,
              "genes_kept": int(keep.sum()), "genes_total": int(len(mean_prof)),
              "kept_fraction_of_mass": float(mean_prof[keep].sum() / max(mean_prof.sum(), 1e-9)),
              "mode": "sampled" if sampling else "mean-rows"}

        if sampling:
            # 从真实 NTC 细胞采样 400/扰动 ⊕ Δ：保留组内方差（官方单细胞 Wilcoxon 需要）
            # 注意: 保留全基因空间（各 context keep 数不同会导致外层 vstack 列数不一致）
            n_ntc = X_ntc.shape[0]
            factor_full = np.ones(len(mean_prof), dtype=np.float32)
            pert_blocks = []
            for g in perts:
                d = dmap.get(g)
                idx = rng.choice(n_ntc, size=N_CELLS_PER_PERT, replace=True)
                blk = X_ntc[idx].copy()
                if d is not None and not d.isna().all():
                    dv = d.reindex(panel).to_numpy()
                    factor = np.power(2.0, cfg["lambda_real"] * np.nan_to_num(dv, nan=0.0)
                                      ).astype(np.float32)
                    blk = blk.multiply(factor.reshape(1, -1)).tocsr()
                blk.data = np.clip(np.rint(blk.data), 0, None).astype(np.float32)
                blk.eliminate_zeros()
                pert_blocks.append(blk)
            blocks.append(sp.vstack(pert_blocks, format="csr"))
        else:
            base = sp.csr_matrix(np.rint(np.where(keep, mean_prof, 0.0)).astype(np.float32).reshape(1, -1))
            pert_rows = []
            for g in perts:
                d = dmap.get(g)
                if d is None or d.isna().all():
                    pert_rows.append(base)
                    continue
                d = d.reindex(panel).to_numpy()
                factor = np.power(2.0, cfg["lambda_real"] * np.nan_to_num(d, nan=0.0))
                row = mean_prof * factor
                row = np.rint(np.clip(row, 0.0, cfg["max_fold"] * np.maximum(mean_prof, 1.0)))
                pert_rows.append(sp.csr_matrix(row.reshape(1, -1)))
            blocks.append(sp.vstack([sp.vstack([r] * N_CELLS_PER_PERT, format="csr")
                                     for r in pert_rows], format="csr"))
        ctx_stats[ctx] = st
        print(f"[ctx {ctx}] {st['mode']} done, genes kept {st['genes_kept']}", flush=True)
        del a_ctx, X_ntc

    X = sp.vstack(blocks).tocsr().astype(np.float32)
    X.eliminate_zeros()
    obs = pd.DataFrame(
        {
            "target_gene": np.tile(np.repeat(np.array(perts), N_CELLS_PER_PERT), len(CONTEXTS)),
            "context": np.repeat(CONTEXTS, N_PERTS * N_CELLS_PER_PERT),
        },
        index=[f"c{c}_p{p}_cell{i}" for c in CONTEXTS for p in range(N_PERTS)
               for i in range(N_CELLS_PER_PERT)],
    )
    a = ad.AnnData(X=X, obs=obs)
    a.var_names = panel.to_numpy()
    a.var.index.name = None
    report["contexts"] = ctx_stats
    return a, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/l1_signature/ws-v1.yaml"))
    args = ap.parse_args()
    cfg = load_config(Path(args.config))

    perts = pd.read_csv(ROOT / cfg["controls_dir"] / cfg["pert_counts_file"])["target_gene"]
    a, report = build_submission(cfg)
    report["validation"] = validate(a, perts)

    out_dir = ROOT / "outputs/submissions"
    stem = cfg["output_stem"]
    a.write_h5ad(out_dir / f"{stem}.h5ad", compression="gzip")
    (ROOT / "outputs/metrics" / f"{stem}.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["validation"], indent=2))
    print(f"[done] {out_dir / f'{stem}.h5ad'}")


if __name__ == "__main__":
    main()
