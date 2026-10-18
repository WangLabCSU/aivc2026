#!/usr/bin/env python
"""hold_sim v5 — 跳出列中心化族 + NC+yh 混合 + 显著性整代表筛选 (iter034 设计臂).

sim4/iter033 教训: NC 保住符号代理 (dF=+0.0000) 但 bench fid 仍 -0.0055 vs v0.7 —
fid = k/max(n_pred, n_conf) 由 pred 侧 DE 显著性门控, 符号保持是必要非充分.
官方 overall = (pds+nmae+fid+reach+jac+0)/6: 可动杠杆 = pds(0.518)/nmae(-0.109)/
fid(-0.030)/reach(0.133). NC 已在 mse/nmae 轴优于 HCX+v0.7 → 本轮沿三条新轴筛选:

  (a) 跳出列中心化族 (结构性): 行中心化 / 双中心 / 基因z-score / 中位数中心化 /
      rank-1 SVD 移除 (跨pert共享成分剥离 — HCX 增益的机制推广)
  (b) NC+yh 混合: 翻转位注入 yh 幅度 (符号锁不变)
  (c) 显著性整代表: 选择性放大 (保号) — 使 |delta| 分布尖锐化, 代理 pred 显著集集中

新增 nmae 代理: truth-DE top-50 上 mean|P-Y| / mean|Y| (越低越好) — 官方 nmae
是第二大杠杆, sim4 未测. gate: S1/S3/F1 沿用 + 候选须 pds d_vs_A ≥ +0.045 且
dF ≥ -0.001.
"""
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
SRC = ["gwps", "hct116_all"]


def load_pairs(name, suffix, hold, col="logfc"):
    tbl = pq.read_table(f"{SIG}/{name}{suffix}.parquet",
                        columns=["target_gene", "gene", col],
                        filters=[("target_gene", "in", sorted(hold))])
    df = tbl.to_pandas()
    g = df.groupby(["target_gene", "gene"], sort=False)[col].mean().reset_index()
    g.columns = ["p", "g", col]
    return g


def de_mask(Y, k=50):
    n, G = Y.shape
    k = min(k, G)
    ath = np.partition(np.abs(Y), -k, axis=1)[:, -k][:, None]
    return np.abs(Y) >= ath


def main():
    t0 = time.time()
    hold = set(json.load(open(f"{ROOT}/data/interim/local_bench/Q_HepG2/targets.json")))
    hold_list = sorted(hold)
    print(f"hold perts={len(hold_list)}", flush=True)

    truth = load_pairs("hepg2_all", "", hold, col="logfc").rename(columns={"logfc": "y"})
    parts = []
    for a in SRC:
        raw = load_pairs(a, "", hold).rename(columns={"logfc": "x"})
        cal = load_pairs(a, "_cal", hold).rename(columns={"logfc": "yh"})
        df = raw.merge(cal, on=["p", "g"], how="inner").merge(truth, on=["p", "g"], how="inner")
        print(f"{a}: paired+truth={len(df):,} t={time.time()-t0:.0f}s", flush=True)
        parts.append(df[["p", "g", "x", "yh", "y"]])
    agg = pd.concat(parts, ignore_index=True).groupby(["p", "g"], sort=False)[
        ["x", "yh", "y"]].mean().reset_index()

    mx = agg.pivot(index="p", columns="g", values="x").reindex(hold_list)
    myh = agg.pivot(index="p", columns="g", values="yh").reindex(hold_list)
    mt = truth.pivot(index="p", columns="g", values="y").reindex(hold_list)
    ok = mx.notna().all(0) & myh.notna().all(0) & mt.notna().all(0)
    ok &= ~pd.Index(ok.index).isin(hold_list)
    ok &= (pd.Index(ok.index) != "non-targeting")
    genes = ok.index[ok.to_numpy()]
    X = mx[genes].to_numpy(dtype=np.float32)
    YH = myh[genes].to_numpy(dtype=np.float32)
    Y = mt[genes].to_numpy(dtype=np.float32)
    print(f"matrix: {X.shape[0]} perts x {X.shape[1]} genes t={time.time()-t0:.0f}s", flush=True)

    n = len(hold_list)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    med = np.median(X, 0, keepdims=True)

    # --- 基础构造 ---
    H = (X - mu).astype(np.float32)                      # HCX (v0.8)
    NC = np.where(X * H > 0, H, X).astype(np.float32)    # NC (v0.11)
    R = (X - X.mean(1, keepdims=True)).astype(np.float32)  # 行中心化 (per-pert)

    cands = {
        "A_x (v0.7)": X,
        "B_yhat (v0.8b)": YH,
        "H_center_x (v0.8)": H,
        "NC_b1.0 (v0.11)": NC,
        # (a) 跳出列中心化族
        "R_rowcenter": R,
        "D_double": (H - H.mean(1, keepdims=True)).astype(np.float32),
        "Z_gene": ((X - mu) / np.maximum(sd, np.median(sd) * 0.1)).astype(np.float32),
        "M_med": (X - med).astype(np.float32),
        "NCmed": np.where(X * (X - med) > 0, X - med, X).astype(np.float32),
    }
    # rank-1/2 SVD 移除 (在中心化基础上, 剥离最大跨pert共享方向)
    for tag, base in (("P1", H), ("P2", R)):
        Ub, Sb, Vt = np.linalg.svd(base.astype(np.float64), full_matrices=False)
        for kk in (1, 2):
            rec = (Ub[:, :kk] * Sb[:kk]) @ Vt[:kk]
            cands[f"{tag}_pc{kk}"] = (base - rec).astype(np.float32)
    # (b) NC+yh 混合 (符号锁)
    flip = X * H <= 0
    yh_ok = (np.sign(YH) == np.sign(X)) & (np.abs(YH) > 1e-6)
    HY = NC.copy()
    HY[flip & yh_ok] = YH[flip & yh_ok]              # 翻转位注入 yh (符号锁)
    cands["HY_flip"] = HY
    cands["W_max"] = np.where(np.abs(YH) > np.abs(X), YH, X).astype(np.float32)
    # (c) 显著性整代表: NC 基础上对大 |v| 基因选择性放大 (保号)
    for g in (1.3, 1.6):
        thr = np.median(np.abs(NC), axis=1, keepdims=True)
        amp = np.where(np.abs(NC) > thr, 1.0 + (g - 1.0), 1.0).astype(np.float32)
        cands[f"AMP_g{g}"] = (NC * amp).astype(np.float32)

    Rn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    DE = de_mask(Y)

    print("\n== sim5: pds + fid 代理 + nmae 代理 ==", flush=True)
    rows = []
    Yab = np.abs(Y)
    for name, P in cands.items():
        Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        D = 1.0 - Pn @ Rn.T
        ranks = rankdata(D, axis=1)
        score = 1.0 - ranks[np.arange(n), np.arange(n)] / n
        agree = (np.sign(P) == np.sign(Y)) & (np.sign(Y) != 0)
        fp = float(np.mean((agree & DE).sum(1) / np.maximum(DE.sum(1), 1)))
        den = float(np.abs(Y)[DE].mean())
        nmae_p = float(np.mean(np.abs((P - Y))[DE & np.isfinite(P)]) / den)
        rows.append((name, float(np.mean(score)), fp, nmae_p))
        print(f"{name:18s} pds={rows[-1][1]:.4f} fidpx={fp:.4f} nmaepx={nmae_p:.4f}", flush=True)

    print("\n== 相对锚点 (A=行0, H=行2, NC=行3) ==", flush=True)
    dA = {r[0]: r[1] - rows[0][1] for r in rows}
    dF = {r[0]: r[2] - rows[0][2] for r in rows}
    dN = {r[0]: r[3] - rows[0][3] for r in rows}
    for r in rows:
        flag = "  <<<" if (dA[r[0]] >= 0.045 and dF[r[0]] >= -0.001) else ""
        print(f"{r[0]:18s} dP={dA[r[0]]:+.4f} dF={dF[r[0]]:+.4f} dNMAE={dN[r[0]]:+.4f}{flag}",
              flush=True)

    print("\n== sanity ==", flush=True)
    d = {r[0]: r[1] for r in rows}
    f = {r[0]: r[2] for r in rows}
    s1 = d["B_yhat (v0.8b)"] < d["A_x (v0.7)"]
    s3 = abs(d["A_x (v0.7)"] - 0.5121) < 0.10
    f1 = f["H_center_x (v0.8)"] < f["A_x (v0.7)"]
    print(f"S1 B<A : {'PASS' if s1 else 'FAIL'}  S3 |A-0.5121|<0.10 : {'PASS' if s3 else 'FAIL'} "
          f"(A={d['A_x (v0.7)']:.4f})", flush=True)
    print(f"F1 fidpx(HCX)<fidpx(A) : {'PASS' if f1 else 'FAIL'} "
          f"(HCX={f['H_center_x (v0.8)']:.4f} A={f['A_x (v0.7)']:.4f})", flush=True)
    print(f"[done] t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
