"""组装 hepg2 全靶 × 全面板特征并输出三臂 Δ̂ 长表 (LOCO-HepG2 交付物)。"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.l15_fusion.common import FeatureBuilder, build_vocab, load_cfg, sig_file, work_dir
from src.models.l15_fusion.predict_loco import predict_mlp, predict_xgb

cfg = load_cfg(); cache = work_dir(cfg)
vocab = build_vocab(cfg)
sig = pd.read_parquet(sig_file("hepg2"), columns=["target_gene", "gene"])
targets = np.array(sorted(sig["target_gene"].unique()), dtype="U")
panel = np.array(sorted(sig["gene"].unique()), dtype="U")
print(f"targets={len(targets)} panel={len(panel)}", flush=True)
TG = np.repeat(targets, len(panel)); GN = np.tile(panel, len(targets))
fb = FeatureBuilder(cfg, vocab, cache)
Xs = []
for s in range(0, len(TG), 2_000_000):
    Xs.append(fb.transform("hepg2", TG[s:s+2_000_000], GN[s:s+2_000_000]))
X = np.concatenate(Xs)
print("features", X.shape, flush=True)
gpos = vocab.get_indexer(pd.Index(GN)).astype(np.int64)
tpos = vocab.get_indexer(pd.Index(TG)).astype(np.int64)
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
px = predict_xgb(cfg, cache, X, 2)
pm = predict_mlp(cfg, cache, X, 2, gpos, tpos, dev, pt_tag="s2_noemb")
pf = (0.5 * px + 0.5 * pm).astype(np.float32)
for name, p in [("xgb", px), ("mlp", pm), ("fused", pf)]:
    d = pd.DataFrame({"target_gene": TG, "gene": GN,
                      "delta": np.clip(np.nan_to_num(p, nan=0.0), -4.0, 4.0).astype(np.float32)})
    d.to_parquet(cache / f"loco_hepg2_full_{name}.parquet")
    print(f"saved {name}: {len(d)} rows", flush=True)
print("full pred complete", flush=True)
