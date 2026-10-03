"""Bizard 预测: 每鸟 Δ̂（bench 五场景 + A/B/C 提交特征, 均复用 v35 特征缓存）。

bench 特征: <bizard>/feat_bench_{ctx}.npz（缺失则现场构建, v35 22 列布局）。
产出: <bizard>/pred_{bird}_{SCTX}.parquet 与 pred_{bird}_submit_{A|B|C}.parquet
  fused = α·xgb + (1−α)·mlp（α=0.5, 对齐 v3.5 提交臂）, clip ±4。

用法: python src/models/bizard/predict.py --bird b1_p [--bench P_H1,Q_HepG2] [--submit]
      python src/models/bizard/predict.py --build-bench-features P_H1,R_Jurkat,S_RPE1,S_HCT116
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.bizard.swarm import (  # noqa: E402
    ROOT, SCENARIO_ORDER, birds, fusion_cfg, load_cfg, workdir)

BENCH = ROOT / "data/interim/local_bench"


def bird_dirs(cfg: dict, bird: dict) -> Path:
    d = workdir(cfg) / bird["name"]
    if bird.get("reuse"):  # 复用 v35 模型目录
        d = ROOT / cfg["v35_workdir"]
    return d


def build_bench_features(cfg: dict, ctxs: list[str], fcfg: dict) -> None:
    """五场景 bench 特征（v35 布局）→ <bizard>/feat_bench_{ctx}.npz。"""
    import anndata as ad
    from src.models.l15_fusion.common import FeatureBuilder, build_vocab
    from src.models.l15_fusion.assemble_data import _chunked_transform
    cache = ROOT / cfg["v35_workdir"]
    out = workdir(cfg)
    vocab = build_vocab(fcfg)
    fb = FeatureBuilder(fcfg, vocab, cache)
    for ctx in ctxs:
        p = out / f"feat_bench_{ctx}.npz"
        if p.exists():
            print(f"feat_bench_{ctx} exists", flush=True)
            continue
        t = time.time()
        targets = json.loads((BENCH / ctx / "targets.json").read_text())
        a = ad.read_h5ad(BENCH / ctx / "ref.h5ad", backed="r")
        panel = np.asarray(a.var_names.astype(str).tolist())
        a.file.close()
        TG = np.repeat(np.array(targets, dtype="U"), len(panel))
        GN = np.tile(panel, len(targets))
        row_ctx = cfg["scenarios"][ctx]["train_ctx"]
        X = _chunked_transform(fb, row_ctx, TG, GN)
        np.savez(p, X=X, target=TG.astype("U"), gene=GN.astype("U"))
        print(f"feat_bench_{ctx}: {X.shape} row_ctx={row_ctx} ({time.time()-t:.0f}s)", flush=True)


def predict_pairs(cfg: dict, fcfg: dict, bird: dict, X: np.ndarray,
                  target: np.ndarray, gene: np.ndarray) -> np.ndarray:
    """单 (鸟, 特征块) → fused Δ̂（xgb+mlp 融合, clip）。复用鸟 → v35 模型 tag。"""
    from src.models.l15_fusion.common import build_vocab
    from src.models.l15_fusion.predict_loco import predict_mlp, predict_xgb
    import torch
    bdir = bird_dirs(cfg, bird)
    tag = bird["name"]
    xgb_tag = bird.get("xgb_tag", tag)
    mlp_tag = bird.get("mlp_tag", tag)
    vocab = build_vocab(fcfg)
    gpos = vocab.get_indexer(pd.Index(gene.astype(str))).astype(np.int64)
    tpos = vocab.get_indexer(pd.Index(target.astype(str))).astype(np.int64)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    px = predict_xgb(fcfg, bdir, X, 2, model_tag=xgb_tag)
    pm = predict_mlp(fcfg, bdir, X, 2, gpos, tpos, dev, pt_tag=mlp_tag)
    alpha = float(cfg["predict"]["alpha"])
    pf = alpha * px + (1 - alpha) * pm
    return np.clip(np.nan_to_num(pf, nan=0.0), -cfg["predict"]["clip"], cfg["predict"]["clip"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bird", default=None)
    ap.add_argument("--bench", default="", help="逗号分隔场景; 空=auto(该鸟 eval 场景)")
    ap.add_argument("--submit", action="store_true", help="同时预测 A/B/C 提交特征")
    ap.add_argument("--build-bench-features", default="", help="仅构建 bench 特征后退出")
    args = ap.parse_args()
    cfg = load_cfg()
    fcfg = fusion_cfg(cfg)
    out = workdir(cfg)

    if args.build_bench_features:
        build_bench_features(cfg, args.build_bench_features.split(","), fcfg)
        return

    bl = next(b for b in birds(cfg) if b["name"] == args.bird)
    if args.bench:
        ctxs = args.bench.split(",")
    else:
        ctxs = ([bl["scenario"]] if bl["scenario"] else []) + ["Q_HepG2"]
        ctxs = list(dict.fromkeys(ctxs))
    for ctx in ctxs:
        assert ctx in SCENARIO_ORDER, ctx
        fp = out / f"pred_{bl['name']}_{ctx}.parquet"
        if fp.exists():
            print(f"pred_{bl['name']}_{ctx} exists, skip", flush=True)
            continue
        t0 = time.time()
        z = np.load(out / f"feat_bench_{ctx}.npz", allow_pickle=False)
        delta = predict_pairs(cfg, fcfg, bl, z["X"], z["target"], z["gene"])
        pd.DataFrame({"target_gene": z["target"].astype(str),
                      "gene": z["gene"].astype(str), "delta": delta}).to_parquet(fp)
        print(f"pred_{bl['name']}_{ctx}: rows={len(delta)} "
              f"absmean={np.abs(delta).mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
        del z
    if args.submit:
        for cx in ["A", "B", "C"]:
            fp = out / f"pred_{bl['name']}_submit_{cx}.parquet"
            if fp.exists():
                print(f"pred_{bl['name']}_submit_{cx} exists, skip", flush=True)
                continue
            t0 = time.time()
            z = np.load(ROOT / cfg["v35_workdir"] / f"feat_submit_{cx}.npz",
                        allow_pickle=False)
            delta = predict_pairs(cfg, fcfg, bl, z["X"], z["target"], z["gene"])
            pd.DataFrame({"target_gene": z["target"].astype(str),
                          "gene": z["gene"].astype(str), "delta": delta}).to_parquet(fp)
            print(f"pred_{bl['name']}_submit_{cx}: rows={len(delta)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del z


if __name__ == "__main__":
    main()
