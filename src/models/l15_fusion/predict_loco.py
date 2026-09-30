"""L1.5 step5: 用训练好的模型对特征缓存做 Δ̂ 预测。

用法: python src/models/l15_fusion/predict_loco.py --stage 1 --cache bench_jurkat \
        --models xgb,mlp --alpha 0.5 --tag s1_jurkat
输出: <workdir>/pred_{tag}_{xgb,mlp,fused}.parquet [target_gene, gene, delta] + .npy 行对齐
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.l15_fusion.common import build_vocab, load_cfg, work_dir


def predict_xgb(cfg, cache, X: np.ndarray, stage: int) -> np.ndarray:
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(str(cache / f"xgb_s{stage}.json"))
    return b.predict(xgb.DMatrix(X))


def predict_mlp(cfg, cache, X: np.ndarray, stage: int, gpos: np.ndarray,
                tpos: np.ndarray, device: str, pt_tag: str | None = None) -> np.ndarray:
    import torch
    tag = pt_tag or f"s{stage}"
    ck = torch.load(cache / f"mlp_{tag}.pt", map_location=device, weights_only=False)
    mu, sd = ck["mu"], ck["sd"]
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xn = (np.where(np.isfinite(X), X, 0.0) - mu) / sd
    E = None
    if ck["ne"] > 0:
        parts = []
        for f in ["emb_se.npy", "emb_tahoe.npy"]:
            p = cache / f
            parts.append(np.load(p).astype(np.float32) if p.exists() else np.zeros(
                (max(tpos.max(), gpos.max()) + 1, ck["ne"] // 2), np.float32))
        E = np.concatenate(parts, axis=1)
    import torch.nn as nn
    nb, ne = ck["names_in"], ck["ne"]
    mc = cfg["mlp"]
    net = nn.Sequential(
        nn.Linear(nb + 2 * ne, mc["hidden"][0]), nn.SiLU(), nn.Dropout(mc["dropout"]),
        nn.Linear(mc["hidden"][0], mc["hidden"][1]), nn.SiLU(), nn.Dropout(mc["dropout"]),
        nn.Linear(mc["hidden"][1], 1)).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    out = np.empty(len(X), np.float32)
    bs = 1 << 20
    with torch.no_grad():
        for s in range(0, len(X), bs):
            x = torch.from_numpy(Xn[s:s + bs]).to(device)
            if E is not None:
                ve = torch.cat([torch.from_numpy(E[gpos[s:s + bs]]).to(device),
                                torch.from_numpy(E[tpos[s:s + bs]]).to(device)], dim=1)
                x = torch.cat([x, ve], dim=1)
            out[s:s + bs] = net(x).squeeze(-1).cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--cache", required=True, help="特征缓存名, 如 bench_hepg2")
    ap.add_argument("--models", default="xgb,mlp")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--clip", type=float, default=4.0)
    ap.add_argument("--mlp-pt", default=None, help="MLP checkpoint tag, 如 s1_noemb")
    args = ap.parse_args()
    cfg = load_cfg()
    cache = work_dir(cfg)
    t0 = time.time()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(cache / f"feat_{args.cache}.npz", allow_pickle=False)
    X = z["X"]
    target = z["target"].astype(str)
    gene = z["gene"].astype(str)
    vocab = build_vocab(cfg)
    gpos = vocab.get_indexer(pd.Index(gene)).astype(np.int64)
    tpos = vocab.get_indexer(pd.Index(target)).astype(np.int64)
    preds = {}
    for m in args.models.split(","):
        t = time.time()
        if m == "xgb":
            preds["xgb"] = predict_xgb(cfg, cache, X, args.stage)
        elif m == "mlp":
            preds["mlp"] = predict_mlp(cfg, cache, X, args.stage, gpos, tpos, device, args.mlp_pt)
        print(f"  pred[{m}] done ({time.time()-t:.0f}s)", flush=True)
    if len(preds) == 2:
        preds["fused"] = (args.alpha * preds["xgb"]
                          + (1 - args.alpha) * preds["mlp"]).astype(np.float32)
    df = pd.DataFrame({"target_gene": target, "gene": gene})
    for m, p in preds.items():
        p = np.clip(np.nan_to_num(p, nan=0.0), -args.clip, args.clip).astype(np.float32)
        df["delta"] = p
        df.to_parquet(cache / f"pred_{args.tag}_{m}.parquet")
        np.save(cache / f"pred_{args.tag}_{m}.npy", p)
    print(f"predict_loco {args.tag} complete ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
