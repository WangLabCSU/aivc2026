"""BizardNet A/B/C Δ̂ 提交准备 (v0.2 skip-head 载体).

LOCO-bench 定案: transformer 门控分量在未见上下文稀释信号 (gate≈0 但输出大方差低相关),
skip 线性证据头 (20 维 fusion 特征直连) 是 LOCO 迁移的最优载体 (λ16=0.3521 vs CCRN 0.3474)。
→ 提交 Δ̂ = full 臂 skip(gen_x), 加 per-context 均值中心化 (家族约定, CCRN 教训)。
用法: python src/models/bizard/bizardnet_submit_deltas.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
sys.path.insert(0, str(ROOT))
from src.models.bizard.bizardnet import (GEN_COLS, ROOT, _std_apply, load_cfg, wd)  # noqa: E402

cfg = load_cfg()
out = wd(cfg)
norm = np.load(out / "train/norm.npz")
ck = torch.load(out / "bizardnet_loco.pt", map_location="cpu", weights_only=False)  # skip 头与臂无关
arm = "full"
try:
    ck = torch.load(out / f"bizardnet_{arm}.pt", map_location="cpu", weights_only=False)
    print(f"using {arm} arm skip head", flush=True)
except FileNotFoundError:
    print(f"full 臂缺失, 回退 loco 臂", flush=True)
W = ck["state"]["skip.weight"].cpu().numpy().ravel()
b = float(ck["state"]["skip.bias"].item())
z = np.load(ROOT / cfg["fusion_v35_dir"] / "feat_train.npz")
names = [str(x) for x in z["names"]]
gen_idx = [i for i, nm in enumerate(names)
           if nm in GEN_COLS or nm.startswith("sig_") or nm.startswith("mod_")]
assert len(gen_idx) == len(W)
stats = {}
for cx in ["A", "B", "C"]:
    zs = np.load(ROOT / cfg["fusion_v35_dir"] / f"feat_submit_{cx}.npz", allow_pickle=True)
    sk = np.clip((_std_apply(zs["X"][:, gen_idx], norm["mu_g"], norm["sd_g"]) @ W + b), -4, 4)
    df = pd.DataFrame({"target_gene": zs["target"].astype(str),
                       "gene": zs["gene"].astype(str), "delta": sk.astype(np.float32)})
    mu, sd_raw = float(df["delta"].mean()), float(df["delta"].std())
    df["delta"] = df["delta"] - mu
    df.to_parquet(out / f"delta_submit_{cx}.parquet")
    stats[cx] = {"raw_mean": mu, "raw_sd": sd_raw, "centered_sd": float(df["delta"].std()),
                 "n": int(len(df)), "n_targets": int(df.target_gene.nunique())}
    print(cx, stats[cx], flush=True)
(out / "submit_delta_stats.json").write_text(json.dumps(
    {"arm": arm, "head": "skip (20d linear evidence head)", "contexts": stats}, indent=1))
print("delta_submit_{A,B,C}.parquet written", flush=True)
