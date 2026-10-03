"""Bizard bench: Δ̂ → 采样载体 pred h5ad → cell-eval2 → bundle anchor 归一化评分。

对齐 run_bench.py 口径（NTC bootstrap 400 细胞 ⊕ 2^(λ·Δ̂), max_fold 8 帽, rint）,
推广到五场景（每场景 control 取 configs/local_bench/contexts.yaml）。
评分: outputs/metrics/local_bench/bundles/<ctx>（五场景 bundle 已建）。

用法: python src/models/bizard/bench_bird.py --tag bz01_b1_p --lam 2.5 \
        --contexts P_H1,Q_HepG2 [--delta-glob "<workdir>/pred_bz01_b1_p_{ctx}.parquet"]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.bizard.swarm import ROOT, SCENARIO_ORDER, load_cfg, save_json, workdir  # noqa: E402

ENV_BIN = Path("/data/home/wsx/miniconda3/envs/aivc2026/bin")
BENCH = ROOT / "data/interim/local_bench"
OUT_DIR = ROOT / "outputs/metrics/local_bench"
CFG = ["--preset", "vcc2026", "--pert-col", "target_gene",
       "--input-type", "counts", "--set", "de.backend=pdex"]


def build_pred(ctx: str, delta: pd.DataFrame, lam: float, outdir: Path, seed: int = 0,
               n_cells: int = 400, max_fold: float = 8.0) -> Path:
    ccfg = load_cfg()["scenarios"][ctx]
    control = ccfg["control"]
    ref = BENCH / ctx / "ref.h5ad"
    targets = json.loads((BENCH / ctx / "targets.json").read_text())
    a = ad.read_h5ad(ref)
    panel = pd.Index(a.var_names.astype(str))
    rng = np.random.default_rng(seed)
    mask = a.obs["target_gene"].astype(str).to_numpy() == control
    Xntc = sp.csr_matrix(a[mask].X)
    del a
    dmap = delta.set_index(["target_gene", "gene"])["delta"]
    pred_perts = [control] + targets
    blocks = []
    for p in pred_perts:
        M = Xntc[rng.choice(Xntc.shape[0], n_cells, replace=True)].copy()
        if p in dmap.index.get_level_values(0):
            dd = dmap.loc[p]
            if isinstance(dd, pd.Series) and isinstance(dd.index, pd.MultiIndex):
                dd = dd.xs(p, level=0)
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
        {"target_gene": np.repeat(pred_perts, n_cells), "context": ctx},
        index=[f"p{i}_c{j}" for i in range(len(pred_perts)) for j in range(n_cells)]))
    pred.var_names = panel.to_numpy()
    pred.var.index.name = None
    outdir.mkdir(parents=True, exist_ok=True)
    pp = outdir / f"{ctx}_pred.h5ad"
    pred.write_h5ad(pp, compression="gzip")
    return pp


def run_celleval(ctx: str, pp: Path, outdir: Path) -> Path:
    ccfg = load_cfg()["scenarios"][ctx]
    cmd = [str(ENV_BIN / "cell-eval2"), "run", "-ap", str(pp), "-ar",
           str(BENCH / ctx / "ref.h5ad"), *CFG, "--control", ccfg["control"],
           "-o", str(outdir / f"{ctx}_eval")]
    print("+", " ".join(cmd), flush=True)
    import subprocess
    subprocess.run(cmd, check=True)
    return outdir / f"{ctx}_eval"


def score_ctx(tag: str, ctx: str) -> dict:
    from src.eval.bench_score import score_tag_ctx
    bundle = OUT_DIR / "bundles" / ctx
    return score_tag_ctx(tag, ctx, bundle if (bundle / "manifest.json").exists() else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--contexts", default="Q_HepG2")
    ap.add_argument("--delta-dir", default=None,
                    help="Δ̂ parquet 目录（默认 bizard workdir）; 文件名含 {ctx}")
    ap.add_argument("--delta-pattern", default="pred_{tagless}_{ctx}.parquet")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_cfg()
    delta_dir = Path(args.delta_dir) if args.delta_dir else workdir(cfg)
    outdir = OUT_DIR / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    tagless = args.tag.replace("bz01_", "")

    per_ctx: dict[str, dict] = {}
    for ctx in args.contexts.split(","):
        assert ctx in SCENARIO_ORDER, ctx
        t0 = time.time()
        if not args.score_only:
            dp = delta_dir / args.delta_pattern.format(tagless=tagless, ctx=ctx)
            if not dp.exists():
                # 聚合体命名回退: agg_delta_{ctx}.parquet
                dp = delta_dir / f"agg_delta_{ctx}.parquet"
            delta = pd.read_parquet(dp)
            pp = build_pred(ctx, delta, args.lam, outdir, seed=args.seed,
                            n_cells=cfg["predict"]["n_cells_per_pert"],
                            max_fold=cfg["predict"]["max_fold"])
            print(f"[{args.tag}/{ctx}] pred built ({time.time()-t0:.0f}s)", flush=True)
            run_celleval(ctx, pp, outdir)
        res = score_ctx(args.tag, ctx)
        per_ctx[ctx] = res
        print(f"[{args.tag}/{ctx}] overall={res['overall']:.4f} "
              f"({time.time()-t0:.0f}s total)", flush=True)

    av = [v["overall"] for v in per_ctx.values() if np.isfinite(v.get("overall", np.nan))]
    summary = {"tag": args.tag, "lam": args.lam,
               "per_context": {c: {"overall": v["overall"],
                                   "per_metric": v["per_metric"],
                                   "bundle_mode": v.get("bundle_mode")}
                                for c, v in per_ctx.items()},
               "mean_overall": float(np.mean(av)) if av else float("nan")}
    save_json(summary, outdir / "bizard_summary.json")
    print(json.dumps({"tag": args.tag, "mean": summary["mean_overall"],
                      "per_ctx": {c: v["overall"] for c, v in per_ctx.items()}},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
