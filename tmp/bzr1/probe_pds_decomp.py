#!/usr/bin/env python
"""probe: 校准器 pds 崩 (0.5520 vs NC 0.8323) 的机理分解 —
(a) 全局幅度收缩 (每靶范数缩水) — 可由 per-target 重标定修复?
(b) top-gene 秩重构 (方向结构真坏) — 只能重训练修复?

用 sim5_cache (X=raw sig, Y=truth, YH=calibrator v2 yhat, 100 靶 × 6146 基因).
pds proxy = mean per-row cosine (与 bench pds_cosine 同构, 尺度不变).
"""
import numpy as np

ROOT = "/data/resource/wanglab/aivc/aivc2026"
z = np.load(f"{ROOT}/tmp/bzr1/sim5_cache.npz", allow_pickle=True)
X, Y, YH = z["X"].astype(np.float64), z["Y"].astype(np.float64), z["YH"].astype(np.float64)
n, G = X.shape

mu = X.mean(0, keepdims=True)
H = X - mu
NC = np.where(X * H > 0, H, X)  # 现行 NC 基座

def cosrow(a, b):
    na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.maximum(na * nb, 1e-9)

def report(name, P):
    c = cosrow(P, Y)
    print(f"{name:28s} pds={c.mean():.4f} med={np.median(c):.4f} q25={np.quantile(c,0.25):.4f}")

print("=== 基线 ===")
report("NC (现行基座)", NC)
report("raw X", X)
report("YH (calibrator v2)", YH)

print("\n=== (a) 全局收缩检验: per-target 重标定 ===")
# 每靶把 YH 线性放大到与 NC 同范数 — 若 pds 恢复到 NC 水平, 崩因=收缩; 若仍 ~0.55-0.6, 崩因=秩重构
sYH = YH * (np.linalg.norm(NC, axis=1, keepdims=True) / np.maximum(np.linalg.norm(YH, axis=1, keepdims=True), 1e-9))
report("YH rescale->|NC|", sYH)

# 逐元素比例中位数 (对符号一致的元素) — 收缩量的直接读数
print(f"norm ratio |YH|/|NC| med={np.median(np.linalg.norm(YH,axis=1)/np.linalg.norm(NC,axis=1)):.4f}")

print("\n=== (b) 秩重构检验 ===")
# NC top-50 处, YH 与 Y 的方向一致率 (分子结构) vs NC 自身
tde = np.abs(Y) >= np.partition(np.abs(Y), -50, axis=1)[:, -50][:, None]
for nm, P in [("NC", NC), ("YH", YH), ("YH rescaled", sYH)]:
    agr = (np.sign(P) == np.sign(Y))[tde].mean()
    print(f"{nm:12s} top50-agree={agr:.4f}")

# spearman 每行 (秩相关) — X vs YH 各自对 Y
from scipy.stats import spearmanr
spX = np.median([spearmanr(X[i], Y[i]).statistic for i in range(n)])
spYH = np.median([spearmanr(YH[i], Y[i]).statistic for i in range(n)])
spNC = np.median([spearmanr(NC[i], Y[i]).statistic for i in range(n)])
print(f"\nspearman med: X={spX:.4f} NC={spNC:.4f} YH={spYH:.4f}")

print("\n=== (c) 混合上限: NC 方向 + YH 只在 NC top-8 处 (即 HY8 构造, 已知最优) vs 反向 ===")
aby_x = np.abs(NC); thx = np.partition(aby_x, -8, axis=1)[:, -8][:, None]
dex = aby_x >= thx
HYx = NC.copy(); sel = dex & (NC * YH > 0)
HYx[sel] = np.sign(NC[sel]) * np.abs(YH[sel])
report("HY8 (NC-DE, |YH| at sel)", HYx)

# 反向: YH 为基座, NC 只供幅度?
HYr = YH.copy(); selr = dex & (NC * YH > 0)
HYr[selr] = np.sign(NC[selr]) * np.abs(NC[selr])
report("YH base + |NC| at NC-DE", HYr)
