"""L1.5 step7: 由 Δ̂ 长表构造 pred h5ad（对齐 local_bench 约定）→ cell-eval2 打分。

用法: python src/models/l15_fusion/run_bench.py --ctx jurkat --pred-tag s1_jurkat_fused \
        --lam 1.5 --bench-tag l15s1_fused_lam1.5 [--score]
ctx: jurkat → data/interim/local_bench/R_Jurkat, hepg2 → Q_HepG2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
ENV_BIN = Path("/data/home/wsx/miniconda3/envs/aivc2026/bin")
BENCH = ROOT / "data/interim/local_bench"
OUT = ROOT / "outputs/metrics/local_bench"
CFG = ["--preset", "vcc2026", "--pert-col", "target_gene", "--control", "non-targeting",
       "--input-type", "counts", "--set", "de.backend=pdex"]


def build_pred(bench_name: str, delta_path: Path, lam: float, outdir: Path,
               seed: int, n_cells: int, max_abs_delta: float, max_fold: float) -> Path:
    bname = {"jurkat": "R_Jurkat", "hepg2": "Q_HepG2"}[bench_name]
    ref = BENCH / bname / "ref.h5ad"
    targets = json.loads((BENCH / bname / "targets.json").read_text())
    a = ad.read_h5ad(ref)
    control = "non-targeting"
    panel = pd.Index(a.var_names.astype(str))
    rng = np.random.default_rng(seed)
    mask = a.obs["target_gene"].astype(str).to_numpy() == control
    Xntc = sp.csr_matrix(a[mask].X)
    del a
    d = pd.read_parquet(delta_path)
    dmap = d.set_index(["target_gene", "gene"])["delta"]
    pred_perts = [control] + targets
    blocks = []
    for p in pred_perts:
        M = Xntc[rng.choice(Xntc.shape[0], n_cells, replace=True)].copy()
        if p in dmap.index.get_level_values(0):
            dd = dmap.loc[p]
            if isinstance(dd, pd.Series) and isinstance(dd.index, pd.MultiIndex):
                dd = dd.xs(p, level=0)  # 保险
            dv = dd.reindex(panel).to_numpy()
            f = np.power(2.0, lam * np.nan_to_num(dv, nan=0.0)).astype(np.float32)
            orig = M.data.copy()
            M.data = np.minimum(M.data * f[M.indices], max_fold * orig)
        M.data = np.clip(np.rint(M.data), 0, None).astype(np.float32)
        M.eliminate_zeros()
        blocks.append(M)
    X = sp.vstack(blocks, format="csr").astype(np.float32)
    X.eliminate_zeros()
    pred = ad.AnnData(X=X, obs=pd.DataFrame(
        {"target_gene": np.repeat(pred_perts, n_cells), "context": bench_name},
        index=[f"p{i}_c{j}" for i in range(len(pred_perts)) for j in range(n_cells)]))
    pred.var_names = panel.to_numpy()
    pred.var.index.name = None
    outdir.mkdir(parents=True, exist_ok=True)
    pp = outdir / f"{bname}_pred.h5ad"
    pred.write_h5ad(pp, compression="gzip")
    return pp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", required=True, choices=["jurkat", "hepg2"])
    ap.add_argument("--pred-tag", required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--bench-tag", required=True)
    ap.add_argument("--max-abs-delta", type=float, default=4.0)
    ap.add_argument("--max-fold", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", action="store_true", help="跑 bench_score 归一化(Q_HepG2)")
    args = ap.parse_args()
    from src.models.l15_fusion.common import load_cfg, work_dir
    cfgd = {"n_cells": 400}
    delta_path = work_dir(load_cfg()) / f"pred_{args.pred_tag}.parquet"
    tag = args.bench_tag
    outdir = OUT / tag
    pp = build_pred(args.ctx, delta_path, args.lam, outdir, args.seed,
                    cfgd["n_cells"], args.max_abs_delta, args.max_fold)
    print(f"pred built: {pp}", flush=True)
    bname = {"jurkat": "R_Jurkat", "hepg2": "Q_HepG2"}[args.ctx]
    ref = BENCH / bname / "ref.h5ad"
    cmd = [str(ENV_BIN / "cell-eval2"), "run", "-ap", str(pp), "-ar", str(ref),
           *CFG, "-o", str(outdir / f"{bname}_eval")]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"cell-eval2 done → {outdir / f'{bname}_eval'}", flush=True)
    if args.score and bname == "Q_HepG2":
        subprocess.run([str(ENV_BIN / "python"), str(ROOT / "src/eval/bench_score.py"),
                        "--tags", tag, "--contexts", "Q_HepG2",
                        "--out", str(outdir / "anchor_normalized.json")], check=True)


if __name__ == "__main__":
    main()
