"""Bizard A/B/C 提交物: 群飞聚合 Δ̂ → 采样载体（λ = 群飞最优）→ validate → h5ad。

对齐 build_submission_v1.py（NTC bootstrap 400 细胞 ⊕ 2^(λ·Δ̂), max_fold 8 帽, rint→csr f32）。

用法: python src/models/bizard/build_submission.py --lam 3.5 \
        --out-stem 20261003_bizard_v01 [--steps build]
之后: TMPDIR=/data/resource/wanglab/aivc/tmpdir vcc prep <h5ad> -g ... --perts ... -o <vcc>
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.bizard.swarm import ROOT, load_cfg  # noqa: E402

CONTROLS = ROOT / "data/raw/controls"
MAX_FOLD = 8.0


def build_submission(cfg: dict, lam: float) -> tuple["ad.AnnData", dict]:
    import anndata as ad
    from src.models.l0_context_mean.make_predictions import (
        CONTEXTS, N_CELLS_PER_PERT, N_PERTS, validate)
    panel = pd.Index(pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"].astype(str))
    perts = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].astype(str).tolist()
    rng = np.random.default_rng(0)
    out = workdir(cfg)
    blocks, ctx_stats = [], {}
    for cx in CONTEXTS:
        t0 = time.time()
        a_ctx = ad.read_h5ad(CONTROLS / f"context_{cx}.h5ad")
        X_ntc = sp.csr_matrix(a_ctx.X)
        del a_ctx
        n_ntc = X_ntc.shape[0]
        lib = np.asarray(X_ntc.sum(axis=1)).ravel()
        d = pd.read_parquet(out / f"agg_delta_submit_{cx}.parquet").set_index(
            ["target_gene", "gene"])["delta"]
        pert_blocks = []
        for g in perts:
            idx = rng.choice(n_ntc, size=N_CELLS_PER_PERT, replace=True)
            blk = X_ntc[idx].copy()
            dd = d.xs(g, level="target_gene", drop_level=True).reindex(panel)
            dv = dd.to_numpy(np.float32)
            if np.isfinite(dv).any() and np.any(dv != 0.0):
                factor = np.power(2.0, lam * np.nan_to_num(dv, nan=0.0)).astype(np.float32)
                orig = blk.data.copy()
                blk.data = np.minimum(blk.data * factor[blk.indices], MAX_FOLD * orig)
            blk.data = np.clip(np.rint(blk.data), 0, None).astype(np.float32)
            blk.eliminate_zeros()
            pert_blocks.append(blk)
        blocks.append(sp.vstack(pert_blocks, format="csr"))
        ctx_stats[cx] = {"n_cells": int(n_ntc), "median_lib_size": float(np.median(lib)),
                         "mode": "sampled_ntc_bootstrap", "lambda": lam,
                         "seconds": round(time.time() - t0, 1)}
        print(f"[ctx {cx}] sampled build done ({time.time()-t0:.0f}s)", flush=True)
        del X_ntc
    X = sp.vstack(blocks).tocsr().astype(np.float32)
    X.eliminate_zeros()
    obs = pd.DataFrame(
        {"target_gene": np.tile(np.repeat(np.array(perts), N_CELLS_PER_PERT), len(CONTEXTS)),
         "context": np.repeat(CONTEXTS, N_PERTS * N_CELLS_PER_PERT)},
        index=[f"c{c}_p{p}_cell{i}" for c in CONTEXTS for p in range(N_PERTS)
               for i in range(N_CELLS_PER_PERT)])
    a = ad.AnnData(X=X, obs=obs)
    a.var_names = panel.to_numpy()
    a.var.index.name = None
    report = {"lambda": lam, "contexts": ctx_stats, "validation": validate(a, pd.Series(perts))}
    return a, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--out-stem", default="20261003_bizard_v01")
    ap.add_argument("--model-label", default="bizard-v0.1 swarm (6 birds, soft-rank avg)")
    args = ap.parse_args()
    cfg = load_cfg()
    a, report = build_submission(cfg, args.lam)
    report["model"] = args.model_label
    report["stem"] = args.out_stem
    out_dir = ROOT / "outputs/submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    hp = out_dir / f"{args.out_stem}.h5ad"
    a.write_h5ad(hp, compression="gzip")
    report["h5ad"] = str(hp)
    print(json.dumps(report["validation"], indent=2), flush=True)
    (ROOT / "outputs/metrics" / f"{args.out_stem}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"[done] {hp}; metrics → outputs/metrics/{args.out_stem}.json", flush=True)


if __name__ == "__main__":
    main()
