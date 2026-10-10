"""Ensemble α0.5 @λ6.5 A/B/C 提交物：Δ̂_ens=0.5·heur+0.5·fusion（skipna 凸组合），载体重建。

Δ̂ 语义与 ens_alpha_scan 位级同源（bench Q 100 靶 local 0.4073@λ4.5 / 0.3181@λ6.5 已证）：
- Δ̂_heur: dualsrc base（gwps×hct116_all 等权均值、单源保留、clip±4、未覆盖靶 gwps generic
  全额兜底）——复用 ens_alpha_scan.heur_delta 逐字逻辑，靶集扩到官方 300。
- Δ̂_fusion: v35 提交侧缓存 delta_submit_{A,B,C}.parquet（s2_35 双模型 α=0.5 融合，
  与 fusion@λ6.5 官方 +0.0545 提交物同源）。
- blend: ens_alpha_scan.blend 逐字（加权 skipna 均值，单源独有对保留单源值）。
- 载体: build_submission_v1.build_submission 逐字（NTC bootstrap 400/pert seed0 ⊕
  2^(6.5·Δ̂_ens)，逐元素 max_fold 8×原始计数帽，rint）——与官方提交物同协议。

用法: python scripts/ens_submit_alpha05_lam65.py [--skip-regression]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
sys.path.insert(0, str(ROOT))
ENV_BIN = Path("/data/home/wsx/miniconda3/envs/aivc2026/bin")

V35 = ROOT / "data/interim/l15_fusion/v35"
SCAN = V35 / "ensemble_scan"
BENCH = ROOT / "data/interim/local_bench/Q_HepG2"
CONTROLS = ROOT / "data/raw/controls"
ALPHA = 0.5
STEM = "20261006_ens_alpha05_lam65"

_spec = importlib.util.spec_from_file_location(
    "ens_alpha_scan", ROOT / "scripts/ens_alpha_scan.py")
ens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ens)


def bench_panel() -> pd.Index:
    import anndata as ad
    _r = ad.read_h5ad(BENCH / "ref.h5ad", backed="r")
    p = pd.Index(_r.var_names.astype(str))
    _r.file.close()
    return p


def regression_check(panel: pd.Index) -> dict:
    """1) blend 复算 delta_a0.5 位级一致（用 bench 面板, 与缓存构建同参）;
    2) bench 面板 ⊆ 提交面板。"""
    bp = bench_panel()
    dh = pd.read_parquet(SCAN / "delta_heur.parquet")
    df = pd.read_parquet(SCAN / "delta_fusion.parquet")
    ref = pd.read_parquet(SCAN / "delta_a0.5.parquet")
    rec = ens.blend(dh, df, 0.5, bp)
    key = ["target_gene", "gene"]
    m = ref.merge(rec, on=key, suffixes=("_c", "_r"), how="outer")
    bit_identical = bool((m["delta_c"].to_numpy() == m["delta_r"].to_numpy()).all())
    subset = bool(bp.isin(panel).all())
    rep = {"blend_recompute_rows": len(rec), "blend_bit_identical_vs_cached": bit_identical,
           "bench_panel_in_submit_panel": subset,
           "bench_panel_size": len(bp), "submit_panel_size": len(panel)}
    print("[regression] " + json.dumps(rep), flush=True)
    if not bit_identical:
        raise SystemExit("regression check FAILED")
    # bench 面板 ⊄ 提交面板为已知面板域差（bench Q 代理面板更窄, 601 基因仅存在于 bench）;
    # 提交物权威面板=官方 18,533, Δ̂ 语义按面板重投影, 仅作信息记录不阻塞。
    return rep


def load_dfs(panel: pd.Index, targets: list[str]) -> tuple[dict, dict]:
    """heur(300 靶, context 无关) × fusion(提交侧缓存, 逐 context) → blend。"""
    t0 = time.time()
    dh = ens.heur_delta(targets, panel)
    rep = {"heur_rows": len(dh),
           "heur_absmean": float(dh.delta.abs().mean())}
    dfs: dict[str, pd.DataFrame] = {}
    for cx in ("A", "B", "C"):
        df = pd.read_parquet(V35 / f"delta_submit_{cx}.parquet")
        assert df["target_gene"].nunique() == len(targets), f"{cx} fusion target set mismatch"
        de = ens.blend(dh, df, ALPHA, panel)
        assert np.isfinite(de["delta"]).all(), f"{cx} non-finite ens delta"
        dfs[cx] = de
        both = (dh.set_index(["target_gene", "gene"]).index
                .intersection(df.set_index(["target_gene", "gene"]).index))
        frac_both = float(de.set_index(["target_gene", "gene"]).index.isin(both).mean())
        rep[f"ens_{cx}_rows"] = len(de)
        rep[f"ens_{cx}_absmean"] = float(de.delta.abs().mean())
        rep[f"ens_{cx}_frac_from_both_sources"] = frac_both
        print(f"[blend {cx}] rows={len(de)} absmean={rep[f'ens_{cx}_absmean']:.4f} "
              f"both_src={frac_both:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return dfs, rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-regression", action="store_true")
    args = ap.parse_args()

    panel = pd.Index(
        pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"].astype(str))
    perts = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].astype(str).tolist()
    print(f"panel={len(panel)} perts={len(perts)}", flush=True)

    report: dict = {"stem": STEM, "alpha": ALPHA, "lam": 6.5,
                    "model": "ensemble 0.5*heuristic(dualsrc base) + 0.5*fusion v3.5 s2_35"}
    if not args.skip_regression:
        report["regression"] = regression_check(panel)

    from src.models.l15_fusion import build_submission_v1 as bs
    bs.LAMBDA = 6.5
    dfs, blend_rep = load_dfs(panel, perts)
    report["blend"] = blend_rep
    t0 = time.time()
    a, sub = bs.build_submission(dfs, STEM)
    val = bs.validate(a, pd.Series(perts))
    report["submission"] = sub
    report["validation"] = val
    out = ROOT / "outputs/submissions" / f"{STEM}.h5ad"
    a.write_h5ad(out, compression="gzip")
    report["h5ad"] = str(out)
    print(json.dumps(val, indent=2), flush=True)
    print(f"[build] h5ad written {out} ({time.time()-t0:.0f}s)", flush=True)
    (ROOT / "outputs/metrics" / f"{STEM}.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"[done] metrics → outputs/metrics/{STEM}.json", flush=True)


if __name__ == "__main__":
    main()
