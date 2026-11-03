#!/usr/bin/env python
"""hold_sim7.py — λ=7 管道等变的 sim: pred lfc_view = clip(v, ±3.807) + 非线性尾.

AMPQ75 惨败根因已明: 管道 pred lfc ≈ λ·v 进入 2^(λv) → fold cap 14 → log2 反推,
等效于 lfc 视图 = min(λ·v / 1?, 3.807)... 精确机制: lfc_pred = log2(2^(λv) capped
at 14×orig) — 对 base≈1 的基因, cap 在 lfc=log2(14)=3.807; 对 base>1 基因 cap 更紧.
sim6 缩全表全败于 fidpx (-0.194 for HT/ST), 证明该代理 fid 对 zeroing 敏感但
真实 bench fid 的 n_pred 机制不同.

本 sim7 引入 pipeline 等变变换 T(v):
  T(v) = sign(v) * min(|λv|, 3.807) / λ  (cap 等变)
再在其上测收缩族 + 微放大族 (幅度 < 帽). 若候选在 T 空间的 fid 排序
与 bench 一致 (NC > AMPQ75 已知), 则 sim7 为可传递的幅度评估器.
"""
import numpy as np
from scipy.stats import rankdata

CACHE = "/data/resource/wanglab/aivc/aivc2026/tmp/bzr1/sim5_cache.npz"


def de_mask(Y, k=50):
    n, G = Y.shape
    k = min(k, G)
    ath = np.partition(np.abs(Y), -k, axis=1)[:, -k][:, None]
    return np.abs(Y) >= ath


def pipeline_view(V, lam=7.0, cap=3.807):
    """近似管道: lfc 空间 cap (元素级). V: (n, G) lfc 签名矩阵."""
    AV = np.abs(V)
    clipped = np.minimum(lam * AV, cap)
    return np.sign(V) * (clipped / lam)


def main():
    z = np.load(CACHE)
    X, Y = z["X"], z["Y"]
    n = len(z["hold"])

    mu = X.mean(0, keepdims=True)
    H = (X - mu).astype(np.float32)
    NC = np.where(X * H > 0, H, X).astype(np.float32)

    Rn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    DE = de_mask(Y)
    Ysign = np.sign(Y)
    diag = np.arange(n)
    Yab = float(np.abs(Y)[DE].mean())

    def evaluate(name, P, rows):
        # 管道等变: 先过 λ cap 再评估 (signature → pred lfc 视图)
        Pv = pipeline_view(P)
        Pn = Pv / np.maximum(np.linalg.norm(Pv, axis=1, keepdims=True), 1e-12)
        D = 1.0 - Pn @ Rn.T
        ranks = rankdata(D, axis=1)
        score = float(np.mean(1.0 - ranks[diag, diag] / n))
        agree = (np.sign(Pv) == Ysign) & (Ysign != 0)
        fp = float(np.mean((agree & DE).sum(1) / np.maximum(DE.sum(1), 1)))
        nmae_p = float(np.mean(np.abs(Pv - Y)[DE & np.isfinite(Pv)]) / Yab)
        rows.append((name, score, fp, nmae_p))

    # truth 也过管道同变换? bench 中 truth 是原始 lfc (hepg2_all 聚合) — 不过帽.
    # 这里保留原始 truth Y, 只对 pred 过帽.
    Ys_cap = Y

    rows = []
    evaluate("A_x", X, rows)
    evaluate("NC", NC, rows)

    # AMPQ75 复现 (q75 ×2.0) — 验证 sim7 能否复现 bench 灾难
    med_nc = np.median(np.abs(NC), axis=1, keepdims=True)
    A = np.abs(NC)
    q75 = np.quantile(A, 0.75, axis=1, keepdims=True)
    amp = np.where(A > q75, 2.0, 1.0).astype(np.float32)
    evaluate("AMP_q75_2.0", (NC * amp).astype(np.float32), rows)

    # 微放大族 (帽内)
    for g in (1.15, 1.3, 1.5):
        evaluate(f"G_uniform_{g}", (NC * g).astype(np.float32), rows)
    amp = np.where(A > med_nc, 1.3, 1.0).astype(np.float32)
    evaluate("G_med_1.3", (NC * amp).astype(np.float32), rows)
    amp = np.where(A > med_nc, 1.5, 1.0).astype(np.float32)
    evaluate("G_med_1.5", (NC * amp).astype(np.float32), rows)
    # 轻收缩族 (帽内, 保守)
    amp = np.where(A <= med_nc, 0.8, 1.0).astype(np.float32)
    evaluate("S_med_0.8", (NC * amp).astype(np.float32), rows)
    evaluate("S_uniform_0.8", (NC * 0.8).astype(np.float32), rows)

    print(f"\n== sim7 全表 (管道等变 λcap) ==", flush=True)
    print(f"{'cand':16s} {'pds':>7s} {'dP':>8s} {'fidpx':>7s} {'dF':>8s} {'nmaepx':>7s} {'dN':>8s}", flush=True)
    for name, p, f, nm in rows:
        dP, dF, dN = p - rows[0][1], f - rows[0][2], nm - rows[0][3]
        print(f"{name:16s} {p:7.4f} {dP:+8.4f} {f:7.4f} {dF:+8.4f} {nm:7.4f} {dN:+8.4f}", flush=True)

    # 关键: AMPQ75 vs NC 在 sim7 的方向是否与 bench 一致 (fid down, nmae down bad)?
    print("\n[check] AMP_q75_2.0 应显示: pds≥NC, fid≤NC, nmae≥NC (与 bench 同向)")


if __name__ == "__main__":
    main()
