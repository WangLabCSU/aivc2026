"""L1.5-fusion v1: 竞赛 A/B/C Δ̂ 预测 + 采样载体提交物构建 + Δ̂ sanity 统计。

流程:
  1) 对每个 context c∈{a,b,c}: 特征 = 300 靶 × (panel∩vocab) 基因, 行 basal 用
     context_{X}.h5ad 真实 NTC 画像; XGB(s2)+MLP(s2_noemb) 融合 (α=0.5), clip ±4 → Δ̂ 长表
     (panel∩vocab 外基因 Δ=0, 不出表)。
  2) Δ̂ sanity: 每靶 |Δ̂|>0.25 显著基因数分布; 与 gwps 签名方向一致率 (|gwps|≥0.25)。
  3) 采样载体: 每 (context, pert) NTC bootstrap 400 细胞 ⊕ 2^(1.5·Δ̂), 逐元素
     max_fold×原始计数帽, rint→csr float32; obs target_gene+context → validate() 全绿。

用法: L15_CFG=configs/l15_fusion/v1-all5.yaml python src/models/l15_fusion/build_submission_v1.py \
        [--steps predict,build] [--out-stem 20261001_l15fusion_v1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
sys.path.insert(0, str(ROOT))
from src.models.l0_context_mean.make_predictions import (  # noqa: E402
    CONTEXTS, N_CELLS_PER_PERT, N_PERTS, validate)
from src.models.l15_fusion.common import (  # noqa: E402
    FeatureBuilder, build_vocab, load_cfg, work_dir)
from src.models.l15_fusion.predict_loco import predict_mlp, predict_xgb  # noqa: E402

CONTROLS = ROOT / "data/raw/controls"
LAMBDA = 1.5
MAX_FOLD = 8.0
ALPHA = 0.5
CLIP = 4.0
CTX_KEY = {"A": "a", "B": "b", "C": "c"}


def predict_abc(cfg, cache, vocab, steps: list[str]) -> dict:
    """对 A/B/C 构建特征并产出融合 Δ̂ 长表（存 v1 workdir）。"""
    import anndata as ad
    panel = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"].astype(str)
    perts = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].astype(str).tolist()
    pg = np.array(sorted(set(panel) & set(vocab)), dtype="U")   # 可预测基因
    TG = np.repeat(np.array(perts, dtype="U"), len(pg))
    GN = np.tile(pg, len(perts))
    gpos = vocab.get_indexer(pd.Index(GN)).astype(np.int64)
    tpos = vocab.get_indexer(pd.Index(TG)).astype(np.int64)
    out = {}
    for cx in CONTEXTS:
        t0 = time.time()
        dp = cache / f"delta_submit_{cx}.parquet"
        if "predict" not in steps and dp.exists():
            out[cx] = pd.read_parquet(dp)   # λ 变体: Δ̂ 缓存直载, 仅重建采样载体
            print(f"[{cx}] delta loaded from cache rows={len(out[cx])}", flush=True)
            continue
        fp = cache / f"feat_submit_{cx}.npz"
        if (fp.exists() or "predict" not in steps) and fp.exists():
            z = np.load(fp, allow_pickle=False)
            X, target, gene = z["X"], z["target"].astype(str), z["gene"].astype(str)
        else:
            fb = FeatureBuilder(cfg, vocab, cache)
            Xs = []
            for s in range(0, len(TG), 2_000_000):
                Xs.append(fb.transform(CTX_KEY[cx], TG[s:s + 2_000_000], GN[s:s + 2_000_000]))
            X = np.concatenate(Xs)
            np.savez(fp, X=X, target=TG.astype("U"), gene=GN.astype("U"))
            target, gene = TG, GN
            print(f"[{cx}] features {X.shape} ({time.time()-t0:.0f}s)", flush=True)
        if "predict" not in steps:
            continue
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        px = predict_xgb(cfg, cache, X, 2)
        pm = predict_mlp(cfg, cache, X, 2, gpos, tpos, dev, pt_tag="s2_noemb")
        pf = (ALPHA * px + (1 - ALPHA) * pm).astype(np.float32)
        df = pd.DataFrame({"target_gene": target, "gene": gene,
                           "delta": np.clip(np.nan_to_num(pf, nan=0.0), -CLIP, CLIP)})
        df.to_parquet(cache / f"delta_submit_{cx}.parquet")
        out[cx] = df
        print(f"[{cx}] delta saved rows={len(df)} sig/靶 median="
              f"{(df.delta.abs() > 0.25).groupby(df.target_gene).sum().median():.0f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del X, px, pm, pf
    return out


def delta_stats(dfs: dict[str, pd.DataFrame]) -> dict:
    """Δ̂ sanity：显著基因数分布 + 与 gwps 签名方向一致率。"""
    gw = pd.read_parquet(ROOT / "data/interim/signatures/gwps.parquet",
                         columns=["target_gene", "gene", "logfc"])
    gw = gw.groupby(["target_gene", "gene"], sort=False)["logfc"].mean()
    res = {"lambda_applied": LAMBDA, "sig_threshold": 0.25, "contexts": {}}
    for cx, df in dfs.items():
        cnt = df.assign(sig=df["delta"].abs() > 0.25).groupby("target_gene")["sig"].sum()
        # 方向一致率: |gwps|≥0.25 且该 (t,g) 有 Δ̂
        m = df.merge(gw.rename("gwps"), left_on=["target_gene", "gene"],
                     right_index=True, how="inner")
        hi = m[m["gwps"].abs() >= 0.25]
        agr = float((np.sign(hi["delta"]) == np.sign(hi["gwps"])).mean())
        per_t = hi.assign(ok=np.sign(hi["delta"]) == np.sign(hi["gwps"])
                          ).groupby("target_gene")["ok"].mean()
        res["contexts"][cx] = {
            "n_rows": int(len(df)),
            "sig_genes_per_target": {"median": float(cnt.median()), "p25": float(cnt.quantile(.25)),
                                     "p75": float(cnt.quantile(.75)), "min": int(cnt.min()),
                                     "max": int(cnt.max()), "mean": float(cnt.mean())},
            "dir_agreement_vs_gwps_hi0.25": agr,
            "dir_agreement_per_target_median": float(per_t.median()) if len(per_t) else None,
            "n_targets_with_gwps": int(hi["target_gene"].nunique()),
            "delta_absmean": float(df["delta"].abs().mean()),
        }
        print(f"[stats {cx}] sig/靶 median={cnt.median():.0f} mean={cnt.mean():.1f} "
              f"[{cnt.min()},{cnt.max()}] dir_agr={agr:.3f}", flush=True)
    return res


def build_submission(dfs: dict[str, pd.DataFrame], stem: str) -> tuple["ad.AnnData", dict]:
    """采样载体提交物：NTC bootstrap ⊕ 2^(λ·Δ̂)，max_fold 帽。"""
    import anndata as ad
    panel = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"].astype(str)
    panel = pd.Index(panel)
    perts = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].astype(str).tolist()
    rng = np.random.default_rng(0)
    blocks, ctx_stats = [], {}
    for cx in CONTEXTS:
        t0 = time.time()
        a_ctx = ad.read_h5ad(CONTROLS / f"context_{cx}.h5ad")
        X_ntc = sp.csr_matrix(a_ctx.X)
        del a_ctx
        n_ntc = X_ntc.shape[0]
        lib = np.asarray(X_ntc.sum(axis=1)).ravel()
        d = dfs[cx].set_index(["target_gene", "gene"])["delta"]
        pert_blocks = []
        for g in perts:
            idx = rng.choice(n_ntc, size=N_CELLS_PER_PERT, replace=True)
            blk = X_ntc[idx].copy()
            dd = d.xs(g, level="target_gene", drop_level=True).reindex(panel)
            dv = dd.to_numpy(np.float32)
            if np.isfinite(dv).any() and np.any(dv != 0.0):
                factor = np.power(2.0, LAMBDA * np.nan_to_num(dv, nan=0.0)).astype(np.float32)
                orig = blk.data.copy()
                blk.data = np.minimum(blk.data * factor[blk.indices], MAX_FOLD * orig)
            blk.data = np.clip(np.rint(blk.data), 0, None).astype(np.float32)
            blk.eliminate_zeros()
            pert_blocks.append(blk)
        blocks.append(sp.vstack(pert_blocks, format="csr"))
        ctx_stats[cx] = {"n_cells": int(n_ntc), "median_lib_size": float(np.median(lib)),
                         "mode": "sampled_ntc_bootstrap", "seconds": round(time.time() - t0, 1)}
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
    return a, {"contexts": ctx_stats}


def main() -> None:
    global LAMBDA
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="predict,stats,build")
    ap.add_argument("--out-stem", default="20261001_l15fusion_v1")
    ap.add_argument("--lam", type=float, default=None,
                    help="应用 λ（默认沿用文件内 LAMBDA=1.5）")
    args = ap.parse_args()
    if args.lam is not None:
        LAMBDA = float(args.lam)
    steps = args.steps.split(",")
    cfg = load_cfg()
    cache = work_dir(cfg)
    vocab = build_vocab(cfg)
    print(f"vocab={len(vocab)} workdir={cache} LAMBDA={LAMBDA}", flush=True)

    dfs = predict_abc(cfg, cache, vocab, steps)
    report = {"stem": args.out_stem, "model": "l15_fusion v1 (all-5 retrain)",
              "alpha": ALPHA, "panel_genes_predicted": None}
    if "stats" in steps:
        report["delta_sanity"] = delta_stats(dfs)
    if "build" in steps:
        a, sub = build_submission(dfs, args.out_stem)
        perts = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].astype(str)
        report["submission"] = sub
        report["validation"] = validate(a, perts)
        out_dir = ROOT / "outputs/submissions"
        out_dir.mkdir(parents=True, exist_ok=True)
        hp = out_dir / f"{args.out_stem}.h5ad"
        a.write_h5ad(hp, compression="gzip")
        report["h5ad"] = str(hp)
        print(json.dumps(report["validation"], indent=2), flush=True)
    (ROOT / "outputs/metrics" / f"{args.out_stem}.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"[done] metrics → outputs/metrics/{args.out_stem}.json", flush=True)


if __name__ == "__main__":
    main()
