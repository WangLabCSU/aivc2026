#!/usr/bin/env python
"""sim8 — fid 机制检验: fid = k/max(n_pred, n_conf) 是 n_pred 的减函数.

已证 (sim7): lfc-cap 等变变换的 fid 代理 dF=0 — 因为 cap 后 |λv| 排序不变,
进显著集的集合不变. 但 AMPQ75 bench fid -0.0324 → 真机制在显著性层面.

sim8 假设: pred lfc = min(λ·v, log2 14) (base>0 基因), Wilcoxon 在 400 cells 上
的显著性由 shift 幅度 × NTC 方差决定. 候选构造改变 v 的幅度谱 → 改变 n_pred
(显著基因数) → fid 变. 错号基因 (NC 不改善方向) 进显著集只增分母 → fid 掉.

用 NTC 400-cell 统计 (ntc_cells_q.npy, 9624 面板基因序) + sim5 cache (6146 列
= 面板基因子集, 序未知 → 用 ntc_mean_g6146.npy 对齐的 6146 维 NTC 统计).

输出: 每候选 {drop8, half8, hy8, amp8, nc, raw} 的
  n_pred_sim (相对), agree_rate_sim, fid_sim, pds_sim, nmae_sim.
"""
import numpy as np
import pandas as pd

ROOT = "/data/resource/wanglab/aivc/aivc2026"
LAM = 7.0
CAP = np.log2(14.0)
KDE = 8

z = np.load(f"{ROOT}/tmp/bzr1/sim5_cache.npz", allow_pickle=True)
X, Y, YH = z["X"], z["Y"], z["YH"]  # (100, 6146)
n, G = X.shape

# ---- NTC 统计 (6146 维) ----
ntc_mean = np.load(f"{ROOT}/tmp/bzr1/ntc_mean_g6146.npy").astype(np.float64)
ntc_sd = np.load(f"{ROOT}/tmp/bzr1/ntc_sd_g6146.npy").astype(np.float64)
cv = ntc_sd / np.maximum(ntc_mean, 0.5)

# ---- 管道响应 ----
def R(v):
    return np.minimum(LAM * v, CAP)

# ---- 显著性代理 ----
# NTC 单细胞 lfc 噪声尺度 (count 空间 log2): Wilcoxon 对 shift 敏感的量级.
# 代理: theta_g = c0 * cv_g (相对噪声), s_g = 0.25*theta_g.
# c0 校准: NC 基准下 n_pred_sim ≈ bench 实测显著基因数量级 (数百/靶).
# 但 sim 内没有逐靶 cells; 用 row 级 (每靶 1 条 v) 代理: 显著 = |λv| > theta.
C0 = 2.5  # 校准常数: 使 NC 下 n_pred 中位数在 ~50-200 量级
theta = C0 * cv + 0.05

def n_pred_sim(v):
    pl = R(v)
    p = 1.0 / (1.0 + np.exp(-(np.abs(pl) - theta) / (0.25 * theta + 1e-6)))
    return p.sum(axis=1)  # per-target

# ---- 候选构造 ----
mu = X.mean(0, keepdims=True)
H = (X - mu).astype(np.float32)
NC = np.where(X * H > 0, H, X).astype(np.float32)

aby = np.abs(YH)
th = np.partition(aby, -KDE, axis=1)[:, -KDE][:, None]  # 第8大的|YH| per row
de = aby >= th
sel = de & (X * YH > 0)

HY = NC.copy();  HY[sel] = np.sign(X[sel]) * aby[sel]
DROP = NC.copy(); DROP[de] = 0.0
HALF = NC.copy(); HALF[de] = NC[de] * 0.5
AMP = NC.copy();  AMP[sel] = np.sign(X[sel]) * np.maximum(aby[sel], np.abs(NC[sel]))

cands = {"raw": X, "nc": NC, "hy8": HY, "drop8": DROP, "half8": HALF, "amp8": AMP}

# ---- bench 指标代理 ----
def truth_de_mask(Y, k=50):
    ath = np.partition(np.abs(Y), -k, axis=1)[:, -k][:, None]
    return np.abs(Y) >= ath

tde = truth_de_mask(Y)

rows = []
for name, v in cands.items():
    pl = R(v)
    npr = n_pred_sim(v)
    # agree rate: 在 truth top-50 DE 处方向一致的比例 (近似 fid 分子)
    agr = (np.sign(pl) == np.sign(Y))[tde].mean()
    # nmae proxy: |pl - Y| / |Y| (capped)
    with np.errstate(divide="ignore", invalid="ignore"):
        nmae = np.minimum(np.abs(pl - Y) / np.maximum(np.abs(Y), 1e-3), 10.0)
    # pds proxy: cosine of pl vs Y
    def cosrow(a, b):
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        return (a * b).sum(1) / np.maximum(na * nb, 1e-9)
    pds = cosrow(np.sign(pl) * np.abs(np.minimum(np.abs(pl), 6.0)),
                 np.sign(Y) * np.abs(np.minimum(np.abs(Y), 6.0))).mean()
    # fid proxy: k / max(n_pred, n_conf); n_conf = truth significant count proxy
    n_conf = (np.abs(Y) > theta).sum(1)
    fid = np.mean(KDE * 50 * 0 + 50.0 / np.maximum(npr, n_conf))
    rows.append(dict(cand=name, n_pred=float(np.median(npr)), n_conf=float(np.median(n_conf)),
                     agree=round(float(agr), 4), pds=round(float(pds), 4),
                     nmae=round(float(nmae.mean()), 4), fid=round(float(fid), 4)))

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(f"{ROOT}/tmp/bzr1/sim8_table.csv", index=False)
print("[done] sim8_table.csv")
