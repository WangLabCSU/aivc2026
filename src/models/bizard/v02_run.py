"""Bizard v0.2 单鸟流水线: 训练(xgb∥mlp) → bench 预测(topk 变异) → 评分 → 适应度。

变异算子（configs/bizard/v02.yaml 每鸟字段）:
  pred_topk   Δ̂ 每靶保留 |Δ̂| 前 K（null=稠密; 截尾不缩量程——区别 L1.7 幅度缩水）
  module_off  特征列置零（训练/预测同掩码）; 当前支持 p53/ifn（mod_ 列）

用法:
  python src/models/bizard/v02_run.py --config configs/bizard/v02.yaml --train --bird b1_p --branch xgb
  python src/models/bizard/v02_run.py --config ... --predict --bird b1_p
  python src/models/bizard/v02_run.py --config ... --bench  --bird b1_p
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
sys.path.insert(0, str(ROOT))


def setup(cfg_path: str):
    import src.models.bizard.swarm as swarm
    swarm.CFG_PATH = ROOT / cfg_path
    cfg = swarm.load_cfg()
    fcfg = swarm.fusion_cfg(cfg)
    nj = (cfg.get("predict") or {}).get("xgb_n_jobs")
    if nj:
        fcfg = json.loads(json.dumps(fcfg))
        fcfg["xgb"]["n_jobs"] = int(nj)
    return swarm, cfg, fcfg


def bird_cfg(swarm, cfg: dict, name: str) -> dict:
    return next(b for b in swarm.birds(cfg) if b["name"] == name)


def module_mask_cols(fcfg: dict, bird: dict) -> list[int]:
    """module_off → 特征列号（FeatureBuilder.names() 布局中 mod_* 列）。"""
    off = bird.get("module_off") or []
    if not off:
        return []
    from src.models.l15_fusion.common import FeatureBuilder, build_vocab
    swarm = sys.modules["src.models.bizard.swarm"]
    cfg = swarm.load_cfg()
    vocab = build_vocab(fcfg)
    names = FeatureBuilder(fcfg, vocab, ROOT / cfg["v35_workdir"]).names()
    return [names.index(f"mod_{m}") for m in off if f"mod_{m}" in names]


def apply_mask(X: np.ndarray, cols: list[int]) -> np.ndarray:
    if not cols:
        return X
    X = X.copy()
    X[:, cols] = 0.0
    return X


def topk_delta(df: pd.DataFrame, k) -> pd.DataFrame:
    """Δ̂ 每靶保留 |Δ̂| 前 K 行（其余丢弃 = factor 1）; k None → 原样。"""
    if k is None:
        return df
    return (df.assign(_a=df["delta"].abs())
              .sort_values("_a", ascending=False, kind="mergesort")
              .groupby("target_gene", sort=False).head(int(k))
              .drop(columns="_a").reset_index(drop=True))


def do_train(cfg_path: str, bird_name: str, branch: str) -> None:
    swarm, cfg, fcfg = setup(cfg_path)
    from src.models.bizard.train_bird import train_mlp_bird, train_xgb_bird
    bird = bird_cfg(swarm, cfg, bird_name)
    outdir = swarm.workdir(cfg) / bird["name"]
    outdir.mkdir(parents=True, exist_ok=True)
    flag = outdir / f".{branch}_done"
    if flag.exists():
        print(f"[{bird_name}/{branch}] done, skip", flush=True)
        return
    t0 = time.time()
    z = np.load(ROOT / cfg["v35_workdir"] / "feat_train.npz", allow_pickle=False)
    X = apply_mask(z["X"], module_mask_cols(fcfg, bird))
    w = sys.modules["src.models.bizard.train_bird"].bird_weights(fcfg, z["ctx"], bird)
    print(f"[{bird_name}/{branch}] feat={X.shape} topk={bird.get('pred_topk')} "
          f"mod_off={bird.get('module_off')} aux={bird.get('aux_weight')}", flush=True)
    if branch == "xgb":
        train_xgb_bird(fcfg, outdir, bird, w, X, z["y"], z["ctx"], t0)
    else:
        train_mlp_bird(fcfg, outdir, bird, w, X, z["y"], z["ctx"], t0)
    flag.write_text(json.dumps({"bird": bird["name"], "branch": branch,
                                "seconds": time.time() - t0}))
    if (outdir / ".xgb_done").exists() and (outdir / ".mlp_done").exists():
        (outdir / ".train_done").write_text('{"gen": "v02"}')


def do_predict(cfg_path: str, bird_name: str) -> None:
    swarm, cfg, fcfg = setup(cfg_path)
    from src.models.bizard.predict import predict_pairs
    bird = bird_cfg(swarm, cfg, bird_name)
    out = swarm.workdir(cfg)
    ctxs = ([bird["scenario"]] if bird.get("scenario") else []) + ["Q_HepG2"]
    for ctx in dict.fromkeys(ctxs):
        fp = out / f"pred_{bird['name']}_{ctx}.parquet"
        if fp.exists():
            print(f"pred_{bird['name']}_{ctx} exists, skip", flush=True)
            continue
        t0 = time.time()
        fp_feat = out / f"feat_bench_{ctx}.npz"
        if not fp_feat.exists():  # 复用 v01 同骨干特征缓存（v35 布局一致）
            fp_feat = ROOT / "data/interim/bizard/v01" / f"feat_bench_{ctx}.npz"
        z = np.load(fp_feat, allow_pickle=False)
        X = apply_mask(z["X"], module_mask_cols(fcfg, bird))
        delta = predict_pairs(cfg, fcfg, bird, X, z["target"], z["gene"])
        df = pd.DataFrame({"target_gene": z["target"].astype(str),
                           "gene": z["gene"].astype(str), "delta": delta})
        df = topk_delta(df, bird.get("pred_topk"))
        df.to_parquet(fp)
        print(f"pred_{bird['name']}_{ctx}: rows={len(df)} (topk={bird.get('pred_topk')}) "
              f"absmean={df['delta'].abs().mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
        del z, X


def do_bench(cfg_path: str, bird_name: str, lam: float) -> None:
    import src.models.bizard.bench_bird as bb
    swarm, cfg, _ = setup(cfg_path)
    bird = bird_cfg(swarm, cfg, bird_name)
    tag = f"bz02_{bird['name']}"
    outdir = ROOT / cfg["bench_out"] / tag
    outdir.mkdir(parents=True, exist_ok=True)
    sm = outdir / "bizard_summary.json"
    if sm.exists():
        print(f"[{tag}] bench exists, skip", flush=True)
        return
    ctxs = ([bird["scenario"]] if bird.get("scenario") else []) + ["Q_HepG2"]
    per_ctx: dict[str, dict] = {}
    for ctx in dict.fromkeys(ctxs):
        t0 = time.time()
        dp = swarm.workdir(cfg) / f"pred_{bird['name']}_{ctx}.parquet"
        delta = pd.read_parquet(dp)
        pp = bb.build_pred(ctx, delta, lam, outdir, seed=0,
                           n_cells=cfg["predict"]["n_cells_per_pert"],
                           max_fold=cfg["predict"]["max_fold"])
        print(f"[{tag}/{ctx}] pred built ({time.time()-t0:.0f}s)", flush=True)
        bb.run_celleval(ctx, pp, outdir)
        res = bb.score_ctx(tag, ctx)
        per_ctx[ctx] = res
        print(f"[{tag}/{ctx}] overall={res['overall']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    fit = cfg["fitness"]
    g = swarm.global_fitness({c: v["overall"] for c, v in per_ctx.items()}, fit)
    raw_vals = [fit["affine_a"] * v["overall"] - fit["affine_b"] for v in per_ctx.values()]
    pen_raw = fit["min_penalty"] * max(0.0, -min(raw_vals))
    rank_raw = float(np.mean(raw_vals)) - pen_raw
    summary = {"tag": tag, "bird": bird["name"], "bird_cfg": bird, "lam": lam,
               "per_ctx": {c: v["overall"] for c, v in per_ctx.items()},
               "per_metric": {c: v["per_metric"] for c, v in per_ctx.items()},
               "global": g, "rank_raw": rank_raw,
               "gen": bird["name"].split("_")[-1] if bird["name"].endswith("_g2") else "g1"}
    swarm.save_json(summary, sm)
    print(json.dumps({"tag": tag, "rank_raw": round(rank_raw, 4),
                      "per_ctx": {c: round(v, 4) for c, v in summary["per_ctx"].items()}},
                     flush=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bizard/v02.yaml")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bird", required=True)
    ap.add_argument("--branch", default="both", choices=["both", "xgb", "mlp"])
    ap.add_argument("--lam", type=float, default=None)
    args = ap.parse_args()
    if args.train:
        for br in (["xgb", "mlp"] if args.branch == "both" else [args.branch]):
            do_train(args.config, args.bird, br)
    if args.predict:
        do_predict(args.config, args.bird)
    if args.bench:
        cfg = setup(args.config)[1]
        do_bench(args.config, args.bird, args.lam or cfg["aggregate"]["bench_lambda"])


if __name__ == "__main__":
    main()
