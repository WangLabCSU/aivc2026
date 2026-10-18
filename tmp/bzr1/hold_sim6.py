#!/usr/bin/env python
"""hold_sim v6 — NC 基础上的弱信号收缩族筛选 (iter034 修正臂).

AMPQ75 bench 惨败根因: ×2.0 放大使 pred 显著集幅度过冲真值 → bench nmae
-2.40 崩盘 (sim5b nmae 代理方向错反 — 放大类构造的幅度代理不可信).
收缩类则 nmae-单调安全: |P|→0 时 |P-Y| 单调不增 (同号过冲收敛/异号趋 |Y|).

候选 (全部 NC 基础, 符号零翻转 → fid 代理必 dF=0):
  S_rowmed_s*   行中位阈值, 弱半区 ×s (s=0.3/0.5/0.7)
  S_q25_s0.5    25分位阈值 (只缩最弱25%)
  HT_med        行中位以下硬归零
  ST_med        软阈值 |v|-τ (τ=行中位)
  WN_med        Wiener v·|v|/(|v|+τ)
pds 代理 + fid 代理筛选, 幅度代理只作参考 (不可信).
"""
import time
import numpy as np
from scipy.stats import rankdata

CACHE = "/data/resource/wanglab/aivc/aivc2026/tmp/bzr1/sim5_cache.npz"


def de_mask(Y, k=50):
    n, G = Y.shape
    k = min(k, G)
    ath = np.partition(np.abs(Y), -k, axis=1)[:, -k][:, None]
    return np.abs(Y) >= ath


def main():
    t0 = time.time()
    z = np.load(CACHE)
    X, Y, hold_list = z["X"], z["Y"], list(z["hold"])
    print(f"[cache] {X.shape}", flush=True)
    n = len(hold_list)

    mu = X.mean(0, keepdims=True)
    H = (X - mu).astype(np.float32)
    NC = np.where(X * H > 0, H, X).astype(np.float32)

    Rn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    DE = de_mask(Y)
    Ysign = np.sign(Y)
    diag = np.arange(n)
    Yab_DE_mean = float(np.abs(Y)[DE].mean())

    def evaluate(name, P, rows):
        Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        D = 1.0 - Pn @ Rn.T
        ranks = rankdata(D, axis=1)
        score = float(np.mean(1.0 - ranks[diag, diag] / n))
        agree = (np.sign(P) == Ysign) & (Ysign != 0)
        fp = float(np.mean((agree & DE).sum(1) / np.maximum(DE.sum(1), 1)))
        nmae_p = float(np.mean(np.abs(P - Y)[DE & np.isfinite(P)]) / Yab_DE_mean)
        rows.append((name, score, fp, nmae_p))

    rows = []
    evaluate("A_x", X, rows)
    evaluate("NC", NC, rows)

    med = np.median(np.abs(NC), axis=1, keepdims=True)
    q25 = np.quantile(np.abs(NC), 0.25, axis=1, keepdims=True)
    A = np.abs(NC)
    for s in (0.3, 0.5, 0.7):
        amp = np.where(A <= med, s, 1.0).astype(np.float32)
        evaluate(f"S_rowmed_s{s}", (NC * amp).astype(np.float32), rows)
    amp = np.where(A <= q25, 0.5, 1.0).astype(np.float32)
    evaluate("S_q25_s0.5", (NC * amp).astype(np.float32), rows)
    evaluate("HT_med", np.where(A <= med, 0.0, NC).astype(np.float32), rows)
    tau = med
    evaluate("ST_med", (np.sign(NC) * np.maximum(A - tau, 0)).astype(np.float32), rows)
    evaluate("WN_med", (NC * A / (A + tau)).astype(np.float32), rows)
    # 混合: 弱半区收缩 + 强半区轻度放大 (AMP教训后保守版)
    amp = np.where(A <= med, 0.5, 1.15).astype(np.float32)
    evaluate("SX_0.5_1.15", (NC * amp).astype(np.float32), rows)

    print(f"\n== sim6 全表 (t={time.time()-t0:.0f}s) ==", flush=True)
    print(f"{'cand':14s} {'pds':>7s} {'dP':>8s} {'fidpx':>7s} {'dF':>8s} {'nmaepx':>7s} {'dN':>8s}", flush=True)
    for name, p, f, nm in rows:
        dP, dF, dN = p - rows[0][1], f - rows[0][2], nm - rows[0][3]
        print(f"{name:14s} {p:7.4f} {dP:+8.4f} {f:7.4f} {dF:+8.4f} {nm:7.4f} {dN:+8.4f}", flush=True)


if __name__ == "__main__":
    main()
