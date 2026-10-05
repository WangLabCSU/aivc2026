"""Bizard v0.3: 表征级变异（混骨干/特征子集/训练行 bootstrap/损失族）+ soft-rank 聚合。

变异维 → 鸟（configs/bizard/v03.yaml）:
  骨干维   heu_dual(dualsrc-deep 启发式) / xgb_s235 / mlp_s235(v35 单臂) / v02_b3r(PBT 鸟)
  特征子集 xgb_sigdrop(去 sig_gwps+hct116) / mlp_sigsub(去 k562+rpe1+jurkat) 互补镜像
  行 bootstrap xgb_rowboot(70% 行, reg:squarederror)
  损失族   mlp_mse(MSE)

用法:
  python src/models/bizard/v03_run.py --build-heuristic
  python src/models/bizard/v03_run.py --train --bird xgb_sigdrop --branch xgb
  python src/models/bizard/v03_run.py --predict --bird B [--submit] [--contexts a,b]
  python src/models/bizard/v03_run.py --correlate
  python src/models/bizard/v03_run.py --aggregate [--submit]
  python src/models/bizard/v03_run.py --bench-agg [--lam 2.5] [--contexts P_H1,...]
  python src/models/bizard/v03_run.py --bench-bird B [--lam 2.5] [--contexts ...]
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

ALL_CTX = ["P_H1", "Q_HepG2", "R_Jurkat", "S_RPE1", "S_HCT116"]


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


def mask_cols(fcfg: dict, cfg: dict, bird: dict) -> list[int]:
    """feature_off（FeatureBuilder 列名）→ 列号。"""
    off = bird.get("feature_off") or []
    if not off:
        return []
    from src.models.l15_fusion.common import FeatureBuilder, build_vocab
    vocab = build_vocab(fcfg)
    names = FeatureBuilder(fcfg, vocab, ROOT / cfg["v35_workdir"]).names()
    return [names.index(c) for c in off if c in names]


def apply_mask(X: np.ndarray, cols: list[int]) -> np.ndarray:
    if not cols:
        return X
    X = X.copy()
    X[:, cols] = 0.0
    return X


def bird_dir(cfg: dict, bird: dict) -> Path:
    base = (bird.get("reuse") or {}).get("base_dir")
    if base == "v35":
        return ROOT / cfg["v35_workdir"]
    if base == "v02":
        return ROOT / "data/interim/bizard/v02" / (bird["reuse"].get("dir") or bird["name"])
    return swarm_workdir(cfg) / bird["name"]


def swarm_workdir(cfg: dict) -> Path:
    import src.models.bizard.swarm as swarm
    return swarm.workdir(cfg)


# ------------------------------------------------------------------ heuristic
def _heu_agg_cache(cfg: dict, out: Path) -> tuple[dict, dict, pd.Series]:
    """每数据集 (target,gene)→mean logfc 聚合缓存（gwps+hct116 parquet 读+groupby ~5min/grid → 只读一次）。

    语义与 make_predictions.load_delta 一致: panel 过滤发生在均值之后（均值与 panel 无关）;
    另缓存 gwps per-gene 全局均值签名（generic 兜底, 仅未覆盖靶使用——本池五场景靶覆盖 100%）。
    """
    import yaml
    sig_dir = ROOT / "data/interim/signatures"
    with open(ROOT / cfg["heuristic_cfg"]) as f:
        hcfg = yaml.safe_load(f)
    parts: dict[str, pd.DataFrame] = {}
    generic: pd.Series | None = None
    for name in hcfg["datasets"]:
        p = out / f"heu_agg_{name}.parquet"
        if p.exists():
            parts[name] = pd.read_parquet(p)
            continue
        df = pd.read_parquet(sig_dir / f"{name}.parquet")
        if name == "gwps":
            generic = df.groupby("gene", sort=False)["logfc"].mean()
        g = df.groupby(["target_gene", "gene"], sort=False)["logfc"].mean().reset_index()
        g.to_parquet(p)
        parts[name] = g
        del df
    if generic is None and (out / "heu_agg_gwps.parquet").exists():
        # generic 缓存缺失时从聚合表退化近似（每靶中logfc→均值, 仅兜底用, 覆盖100%时不用）
        gz = parts["gwps"]
        generic = gz.groupby("gene", sort=False)["logfc"].mean()
    out.mkdir(parents=True, exist_ok=True)
    return hcfg, parts, generic


def heuristic_delta(cfg: dict, target: np.ndarray, gene: np.ndarray,
                    cache: tuple | None = None) -> np.ndarray:
    """dualsrc-deep 启发式 Δ 映射到稠密网格（缺行=0, 语义=无证据不改）。"""
    out = swarm_workdir(cfg)
    hcfg, parts, generic = cache if cache is not None else _heu_agg_cache(cfg, out)
    t0 = time.time()
    panel = set(np.unique(gene.astype(str)).tolist())
    targets = list(np.unique(target.astype(str)))
    tset = set(targets)
    covered: set = set()
    parts_f = []
    for name, g in parts.items():
        gf = g[g["target_gene"].isin(tset) & g["gene"].isin(panel)]
        covered |= set(gf["target_gene"])
        parts_f.append(gf.rename(columns={"logfc": "delta"})[
            ["target_gene", "gene", "delta"]])
    real = (pd.concat(parts_f, ignore_index=True).groupby(
        ["target_gene", "gene"], sort=False)["delta"].mean().reset_index()
        if parts_f else pd.DataFrame(columns=["target_gene", "gene", "delta"]))
    uncovered = [t for t in targets if t not in covered]
    if uncovered and generic is not None and float(hcfg.get("lambda_generic") or 0) > 0:
        gen = generic[generic.index.astype(str).isin(panel)]
        real = pd.concat([real, pd.DataFrame({
            "target_gene": np.repeat(uncovered, len(gen)),
            "gene": np.tile(gen.index.astype(str).to_numpy(), len(uncovered)),
            "delta": np.tile(gen.to_numpy(), len(uncovered))})], ignore_index=True)
    clip = float(hcfg["max_abs_delta"])
    real["delta"] = real["delta"].clip(-clip, clip)
    key = pd.MultiIndex.from_arrays([target.astype(str), gene.astype(str)])
    dkey = pd.MultiIndex.from_arrays([real["target_gene"].astype(str),
                                      real["gene"].astype(str)])
    dser = pd.Series(real["delta"].to_numpy(np.float64), index=dkey)
    dser = dser[~dser.index.duplicated(keep="first")]
    dv = dser.reindex(key).to_numpy(np.float32)
    dv = np.nan_to_num(dv, nan=0.0)
    print(f"[heu_dual] grid={len(key)} covered_rows={int((dv != 0).sum())} "
          f"delta_rows={len(real)} ({time.time()-t0:.0f}s)", flush=True)
    return dv.astype(np.float32)


def do_build_heuristic(cfg_path: str) -> None:
    swarm, cfg, _ = setup(cfg_path)
    out = swarm_workdir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    cache = _heu_agg_cache(cfg, out)   # 一次性建缓存（~3-5 min）
    jobs = [(c, out / f"feat_bench_{c}.npz",
             out / f"pred_heu_dual_{c}.parquet") for c in ALL_CTX]
    jobs += [(f"submit_{c}", ROOT / cfg["v35_workdir"] / f"feat_submit_{c}.npz",
              out / f"pred_heu_dual_submit_{c}.parquet") for c in ("A", "B", "C")]
    for ctx, fp_feat, fp_out in jobs:
        if fp_out.exists():
            print(f"{fp_out.name} exists, skip", flush=True)
            continue
        if not fp_feat.exists():  # bench 特征回退 v01 缓存（v35 布局一致）
            fp_feat = ROOT / "data/interim/bizard/v01" / fp_feat.name
        z = np.load(fp_feat, allow_pickle=False)
        dv = heuristic_delta(cfg, z["target"], z["gene"], cache=cache)
        pd.DataFrame({"target_gene": z["target"].astype(str),
                      "gene": z["gene"].astype(str), "delta": dv}).to_parquet(fp_out)
        del z


# ---------------------------------------------------------------------- train
def do_train(cfg_path: str, bird_name: str, branch: str) -> None:
    swarm, cfg, fcfg = setup(cfg_path)
    from src.models.bizard.train_bird import train_mlp_bird, train_xgb_bird
    bird = bird_cfg(swarm, cfg, bird_name)
    outdir = swarm_workdir(cfg) / bird["name"]
    outdir.mkdir(parents=True, exist_ok=True)
    flag = outdir / f".{branch}_done"
    if flag.exists():
        print(f"[{bird_name}/{branch}] done, skip", flush=True)
        return
    t0 = time.time()
    z = np.load(ROOT / cfg["v35_workdir"] / "feat_train.npz", allow_pickle=False)
    X, y, ctx = apply_mask(z["X"], mask_cols(fcfg, cfg, bird)), z["y"], z["ctx"]
    rf = bird.get("row_frac")
    if rf and rf < 1.0:   # 训练行 bootstrap（行 bootstrap 维）
        rng = np.random.default_rng(int(bird["seed"]) + 100)
        keep = rng.random(len(y)) < float(rf)
        X, y, ctx = X[keep], y[keep], ctx[keep]
        print(f"[{bird_name}] row bootstrap {rf}: rows={keep.sum()}/{len(keep)}", flush=True)
    w = sys.modules["src.models.bizard.train_bird"].bird_weights(fcfg, ctx, bird)
    print(f"[{bird_name}/{branch}] feat={X.shape} off={bird.get('feature_off')} "
          f"loss={bird.get('mlp_loss') or bird.get('xgb_objective') or 'default'} "
          f"aux={bird.get('aux_weight')}", flush=True)
    if branch == "xgb":
        train_xgb_bird(fcfg, outdir, bird, w, X, y, ctx, t0)
    else:
        train_mlp_bird(fcfg, outdir, bird, w, X, y, ctx, t0)
    flag.write_text(json.dumps({"bird": bird_name, "branch": branch,
                                "seconds": time.time() - t0}))


# -------------------------------------------------------------------- predict
def do_predict(cfg_path: str, bird_name: str, contexts: list[str] | None = None,
               submit: bool = False) -> None:
    swarm, cfg, fcfg = setup(cfg_path)
    from src.models.l15_fusion.common import build_vocab
    from src.models.l15_fusion.predict_loco import predict_mlp, predict_xgb
    bird = bird_cfg(swarm, cfg, bird_name)
    out = swarm_workdir(cfg)
    cols = mask_cols(fcfg, cfg, bird)
    bdir = bird_dir(cfg, bird)
    alpha = float(bird.get("alpha", cfg["predict"]["alpha"]))
    xgb_tag = bird.get("xgb_tag", bird_name)
    mlp_tag = bird.get("mlp_tag", bird_name)
    ctxs = list(contexts) if contexts else ALL_CTX
    if submit:
        ctxs += [f"submit_{c}" for c in ("A", "B", "C")]
    vocab = build_vocab(fcfg)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for ctx in dict.fromkeys(ctxs):
        fp = out / f"pred_{bird_name}_{ctx}.parquet"
        if fp.exists():
            print(f"pred_{bird_name}_{ctx} exists, skip", flush=True)
            continue
        t0 = time.time()
        if ctx.startswith("submit_"):
            fp_feat = ROOT / cfg["v35_workdir"] / f"feat_{ctx}.npz"
        else:
            fp_feat = out / f"feat_bench_{ctx}.npz"
            if not fp_feat.exists():
                fp_feat = ROOT / "data/interim/bizard/v01" / fp_feat.name
        z = np.load(fp_feat, allow_pickle=False)
        if bird.get("kind") == "heuristic":
            delta = heuristic_delta(cfg, z["target"], z["gene"])
        else:
            X = apply_mask(z["X"], cols)
            px = pm = None
            if alpha > 0:
                px = predict_xgb(fcfg, bdir, X, 2, model_tag=xgb_tag)
            if alpha < 1:
                gpos = vocab.get_indexer(pd.Index(z["gene"].astype(str))).astype(np.int64)
                tpos = vocab.get_indexer(pd.Index(z["target"].astype(str))).astype(np.int64)
                pm = predict_mlp(fcfg, bdir, X, 2, gpos, tpos, dev, pt_tag=mlp_tag)
            pf = px if pm is None else (pm if px is None else alpha * px + (1 - alpha) * pm)
            delta = np.clip(np.nan_to_num(pf, nan=0.0),
                            -cfg["predict"]["clip"], cfg["predict"]["clip"]).astype(np.float32)
            del X
        pd.DataFrame({"target_gene": z["target"].astype(str),
                      "gene": z["gene"].astype(str), "delta": delta}).to_parquet(fp)
        print(f"pred_{bird_name}_{ctx}: rows={len(delta)} absmean={np.abs(delta).mean():.4f} "
              f"alpha={alpha} ({time.time()-t0:.0f}s)", flush=True)
        del z


# ----------------------------------------------------------------- correlate
def _load_pred(out: Path, bird: str, ctx: str) -> np.ndarray:
    return pd.read_parquet(out / f"pred_{bird}_{ctx}.parquet")["delta"].to_numpy(np.float32)


def do_correlate(cfg_path: str, contexts: list[str]) -> dict:
    swarm, cfg, _ = setup(cfg_path)
    names = [b["name"] for b in swarm.birds(cfg)]
    out = swarm_workdir(cfg)
    report: dict = {}
    for ctx in contexts:
        have = [b for b in names if (out / f"pred_{b}_{ctx}.parquet").exists()]
        if len(have) < 2:
            continue
        P = np.stack([_load_pred(out, b, ctx) for b in have])
        C = np.corrcoef(P)
        report[ctx] = {"birds": have, "matrix": C.round(4).tolist(),
                       "mean_pairwise": float((C.sum() - len(have)) / (len(have)**2 - len(have))),
                       "max_pairwise": float(np.max(C - np.eye(len(have))))}
        print(f"[corr {ctx}] n={len(have)} mean={report[ctx]['mean_pairwise']:.4f} "
              f"max={report[ctx]['max_pairwise']:.4f}", flush=True)
    (out / "bird_correlation.json").write_text(json.dumps(report, indent=2))
    return report


# ------------------------------------------------------------------ aggregate
def do_aggregate(cfg_path: str, bench_ctxs: list[str], submit: bool) -> None:
    swarm, cfg, _ = setup(cfg_path)
    out = swarm_workdir(cfg)
    names = [b["name"] for b in swarm.birds(cfg)]
    ctxs = list(bench_ctxs) + ([f"submit_{c}" for c in ("A", "B", "C")] if submit else [])
    for ctx in ctxs:
        fp_out = out / f"agg_delta_{ctx}.parquet"
        if fp_out.exists():
            print(f"{fp_out.name} exists, skip", flush=True)
            continue
        paths = [out / f"pred_{b}_{ctx}.parquet" for b in names]
        have = [n for n, p in zip(names, paths) if p.exists()]
        dfs = [pd.read_parquet(p) for p in paths if p.exists()]
        base = dfs[0][["target_gene", "gene"]]
        preds = []
        for df in dfs:   # 稠密网格严格对齐（全部鸟产自同一 feat 网格）
            assert len(df) == len(base) \
                and (df["target_gene"].to_numpy() == base["target_gene"].to_numpy()).all() \
                and (df["gene"].to_numpy() == base["gene"].to_numpy()).all(), "鸟间网格不对齐"
            preds.append(df["delta"].to_numpy(np.float32))
        agg = swarm.aggregate(preds, method=cfg["aggregate"]["method"])
        pd.DataFrame({"target_gene": base["target_gene"], "gene": base["gene"],
                      "delta": agg}).to_parquet(fp_out)
        st = {"n_birds": len(preds), "birds": have, "method": cfg["aggregate"]["method"],
              "absmean_agg": float(np.abs(agg).mean()),
              "pearson_vs_birdmean": float(np.corrcoef(
                  np.mean(preds, axis=0), agg)[0, 1])}
        print(f"agg {ctx}: {json.dumps(st)}", flush=True)


# ---------------------------------------------------------------------- bench
def do_bench(cfg_path: str, tag: str, ctxs: list[str], lam: float) -> None:
    import src.models.bizard.bench_bird as bb
    swarm, cfg, _ = setup(cfg_path)
    outdir = ROOT / cfg["bench_out"] / tag
    outdir.mkdir(parents=True, exist_ok=True)
    per_ctx: dict[str, float] = {}
    per_metric: dict[str, dict] = {}
    sm = outdir / "bizard_summary.json"
    if sm.exists():   # 断点续评: 载入已完成场景, 只补缺失场景
        prev = json.loads(sm.read_text())
        per_ctx.update(prev.get("per_ctx") or {})
        per_metric.update(prev.get("per_metric") or {})
    for ctx in ctxs:
        if ctx in per_ctx:
            print(f"[{tag}/{ctx}] already scored={per_ctx[ctx]:.4f}, skip", flush=True)
            continue
        t0 = time.time()
        delta = pd.read_parquet(swarm_workdir(cfg) / f"agg_delta_{ctx}.parquet")
        pp = bb.build_pred(ctx, delta, lam, outdir, seed=0,
                           n_cells=cfg["predict"]["n_cells_per_pert"],
                           max_fold=cfg["predict"]["max_fold"])
        print(f"[{tag}/{ctx}] pred built ({time.time()-t0:.0f}s)", flush=True)
        bb.run_celleval(ctx, pp, outdir)
        res = bb.score_ctx(tag, ctx)
        per_ctx[ctx] = res["overall"]
        per_metric[ctx] = res.get("per_metric") or {}
        print(f"[{tag}/{ctx}] overall={res['overall']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        swarm.save_json({"tag": tag, "lam": lam,
                         "per_ctx": {c: v for c, v in per_ctx.items()},
                         "per_metric": per_metric}, sm)
    print(json.dumps({"tag": tag, "per_ctx": per_ctx}), flush=True)


def do_bench_bird(cfg_path: str, bird_name: str, ctxs: list[str], lam: float) -> None:
    import src.models.bizard.bench_bird as bb
    swarm, cfg, _ = setup(cfg_path)
    tag = f"bz03_{bird_name}"
    outdir = ROOT / cfg["bench_out"] / tag
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "bizard_summary.json").exists():
        print(f"[{tag}] bench exists, skip", flush=True)
        return
    per_ctx: dict[str, dict] = {}
    for ctx in ctxs:
        t0 = time.time()
        delta = pd.read_parquet(swarm_workdir(cfg) / f"pred_{bird_name}_{ctx}.parquet")
        pp = bb.build_pred(ctx, delta, lam, outdir, seed=0,
                           n_cells=cfg["predict"]["n_cells_per_pert"],
                           max_fold=cfg["predict"]["max_fold"])
        bb.run_celleval(ctx, pp, outdir)
        per_ctx[ctx] = bb.score_ctx(tag, ctx)
        print(f"[{tag}/{ctx}] overall={per_ctx[ctx]['overall']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    swarm.save_json({"tag": tag, "bird": bird_name, "lam": lam,
                     "per_ctx": {c: v["overall"] for c, v in per_ctx.items()},
                     "per_metric": {c: v.get("per_metric") for c, v in per_ctx.items()}},
                    outdir / "bizard_summary.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bizard/v03.yaml")
    ap.add_argument("--build-heuristic", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--correlate", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--bench-agg", action="store_true")
    ap.add_argument("--bench-bird", default=None)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--bird", default=None)
    ap.add_argument("--branch", default="both", choices=["both", "xgb", "mlp"])
    ap.add_argument("--contexts", default=None)
    ap.add_argument("--lam", type=float, default=None)
    args = ap.parse_args()
    cfg = setup(args.config)[1]
    ctxs = args.contexts.split(",") if args.contexts else None
    if args.build_heuristic:
        do_build_heuristic(args.config)
    if args.train:
        for br in (["xgb", "mlp"] if args.branch == "both" else [args.branch]):
            do_train(args.config, args.bird, br)
    if args.predict:
        do_predict(args.config, args.bird, ctxs, submit=args.submit)
    if args.correlate:
        do_correlate(args.config, ctxs or ALL_CTX)
    if args.aggregate:
        do_aggregate(args.config, ctxs or ALL_CTX, submit=args.submit)
    if args.bench_agg:
        lam = args.lam or cfg["aggregate"]["bench_lambda"]
        do_bench(args.config, f"bz03_agg_lam{lam:g}", ctxs or ALL_CTX, lam)
    if args.bench_bird:
        lam = args.lam or cfg["aggregate"]["bench_lambda"]
        do_bench_bird(args.config, args.bench_bird, ctxs or ALL_CTX, lam)


if __name__ == "__main__":
    main()
