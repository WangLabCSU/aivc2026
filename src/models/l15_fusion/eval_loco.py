"""L1.5 step6: LOCO proxy 评估 — 学习式 vs 均匀 λ 启发式（gwps 签名）。

对 proxy 缓存（每靶 top200|true| + 300 随机基因）计算:
  spearman_med   — 每靶 Spearman(pred, true) 中位数（≥50 基因的靶）
  recall@100     — true top-100 |logfc| 基因被 pred top-100 覆盖比例（均值）
  sign_acc       — |true|≥0.5 基因上的方向一致率
  pearson_top    — |true|≥0.5 基因上的 Pearson

用法: python src/models/l15_fusion/eval_loco.py --context jurkat \
        --preds pred_s1_jurkat_xgb,pred_s1_jurkat_mlp,pred_s1_jurkat_fused
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.l15_fusion.common import build_sig_store, build_vocab, load_cfg, sig_lookup, work_dir


def metrics(true: np.ndarray, pred: np.ndarray, target: np.ndarray) -> dict:
    df = pd.DataFrame({"t": target, "y": true, "p": pred})
    sps, rec = [], []
    for tg, sub in df.groupby("t", sort=False):
        if len(sub) < 50:
            continue
        if sub["p"].std() == 0 or sub["y"].std() == 0:
            continue
        sps.append(sub["y"].corr(sub["p"], method="spearman"))
        k = min(100, len(sub))
        top_t = set(np.argsort(-np.abs(sub["y"].to_numpy()))[:k])
        top_p = set(np.argsort(-np.abs(sub["p"].to_numpy()))[:k])
        rec.append(len(top_t & top_p) / k)
    hi = df[np.abs(df["y"]) >= 0.5]
    sign_acc = float((np.sign(hi["y"]) == np.sign(hi["p"])).mean()) if len(hi) else np.nan
    pear = float(hi["y"].corr(hi["p"])) if len(hi) > 10 else np.nan
    return {"spearman_med": float(np.median(sps)), "spearman_n": len(sps),
            "recall@100": float(np.mean(rec)), "sign_acc": sign_acc,
            "pearson_hi": pear}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True)
    ap.add_argument("--preds", required=True, help="逗号分隔 pred tag 前缀")
    args = ap.parse_args()
    cfg = load_cfg()
    cache = work_dir(cfg)
    z = np.load(cache / f"feat_proxy_{args.context}.npz", allow_pickle=False)
    true, target = z["y"], z["target"].astype(str)
    vocab = build_vocab(cfg)
    gpos = vocab.get_indexer(pd.Index(z["gene"].astype(str)))
    res = {"context": args.context, "heuristic_gwps":
           metrics(true, np.nan_to_num(sig_lookup(build_sig_store(
               "gwps", vocab, cache), target.astype("U"), gpos.astype(np.int64)), nan=0.0), target)}
    for p in args.preds.split(","):
        pred = np.load(cache / f"{p}.npy")
        res[p] = metrics(true, pred, target)
    (cache / f"eval_proxy_{args.context}.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, {m: round(x, 4) for m, x in v.items() if m != "spearman_n"}, flush=True)


if __name__ == "__main__":
    main()
