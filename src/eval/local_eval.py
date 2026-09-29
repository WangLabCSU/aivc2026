"""本地评测（提交 gate）：用 VCC2025 test split 做 H1 零样本自测。

对 test 的每个扰动：pred = NTC均值 ⊕ λ·Δ(gwps签名→18,080空间)；
L1 与 L0 锚点两种预测分别过 cell-eval2（--preset vcc2026），对比后决定是否消耗线上额度。

用法: python src/eval/local_eval.py [--lambda-real 0.5] [--skip-l0]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
ENV_BIN = Path("/data/home/wsx/miniconda3/envs/aivc2026/bin")
REAL = ROOT / "data/public/vcc2025/adata_Test.h5ad"
SIG = ROOT / "data/interim/signatures/gwps.parquet"
OUTBASE = ROOT / "outputs/metrics/local"
CONTROL = "non-targeting"
N_CELLS = 400


def ntc_mean_profile(a: ad.AnnData) -> np.ndarray:
    mask = a.obs["target_gene"].astype(str).to_numpy() == CONTROL
    X = sp.csr_matrix(a[mask].X)
    mean = np.asarray(X.mean(axis=0)).ravel()
    return np.rint(mean).astype(np.float64)


def build_pred(a: ad.AnnData, mean: np.ndarray, dmap: pd.Series | None,
               lam: float, max_fold: float = 8.0) -> ad.AnnData:
    tg = a.obs["target_gene"].astype(str)
    perts = [p for p in tg.unique() if p != CONTROL]
    panel = pd.Index(a.var_names.astype(str))
    blocks = []
    for p in perts:
        row = mean.copy()
        if dmap is not None and p in dmap.index.get_level_values(0):
            d = dmap.loc[p]
            d = d.reindex(panel).to_numpy()
            factor = np.power(2.0, lam * np.nan_to_num(d, nan=0.0))
            row = np.rint(np.clip(mean * factor, 0.0, max_fold * np.maximum(mean, 1.0)))
        blocks.append(sp.csr_matrix(np.tile(row, (N_CELLS, 1))))
    X = sp.vstack(blocks, format="csr").astype(np.float32)
    X.eliminate_zeros()
    obs = pd.DataFrame(
        {"target_gene": np.repeat(perts, N_CELLS), "context": "T"},
        index=[f"p{p}_c{i}" for p in range(len(perts)) for i in range(N_CELLS)],
    )
    out = ad.AnnData(X=X, obs=obs)
    out.var_names = panel.to_numpy()
    out.var.index.name = None
    return out


def run_celleval(pred: Path, tag: str) -> dict:
    outdir = OUTBASE / tag
    cmd = [str(ENV_BIN / "cell-eval2"), "run", "-ap", str(pred), "-ar", str(REAL),
           "--preset", "vcc2026", "--pert-col", "target_gene", "--control", CONTROL,
           "--input-type", "counts", "-o", str(outdir)]
    print("[run]", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    df = pd.read_csv(outdir / "results.csv")
    return df.to_dict(orient="records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-real", type=float, default=0.5)
    ap.add_argument("--skip-l0", action="store_true")
    args = ap.parse_args()
    OUTBASE.mkdir(parents=True, exist_ok=True)

    a = ad.read_h5ad(REAL)  # 170k×18k，内存可承受
    tg = a.obs["target_gene"].astype(str)
    perts = [p for p in tg.unique() if p != CONTROL]
    mean = ntc_mean_profile(a)
    panel = pd.Index(a.var_names.astype(str))

    sig = pd.read_parquet(SIG)
    sig = sig[sig["gene"].isin(panel)]
    sig_g = sig.groupby(["target_gene", "gene"], sort=False)["logfc"].mean()
    hits = len(set(perts) & set(sig.index.get_level_values(0)))
    print(f"[info] test perts={len(perts)}, gwps coverage={hits}", flush=True)

    results = {"lambda_real": args.lambda_real, "n_test_perts": len(perts),
               "gwps_coverage": hits}

    # L1
    pred = build_pred(a, mean, sig_g, args.lambda_real)
    p1 = OUTBASE / f"l1_lam{args.lambda_real}.h5ad"
    pred.write_h5ad(p1, compression="gzip")
    results["l1"] = run_celleval(p1, f"l1_lam{args.lambda_real}")

    # L0 锚点
    if not args.skip_l0:
        pred0 = build_pred(a, mean, None, 0.0)
        p0 = OUTBASE / "l0_anchor.h5ad"
        pred0.write_h5ad(p0, compression="gzip")
        results["l0_anchor"] = run_celleval(p0, "l0_anchor")

    (OUTBASE / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
