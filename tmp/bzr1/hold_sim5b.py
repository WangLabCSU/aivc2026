#!/usr/bin/env python
"""hold_sim v5b — AMP 家族参数面细扫 (iter034).

sim5 结论: AMP_g1.6 (NC 基础上对 |v|>行中位数 选择性放大, 保号) 是首个
pds+nmae 同时改善且 fid 代理零损的候选 (dP+0.0504/dF+0.0000/dNMAE-0.0159).
fid = k/max(n_pred,n_conf) 机制下, 放大 = 把符号自信基因推深进 pred 显著集
(n_pred 不缩), 是合法 lever; 缩小半区会压 n_pred 触发 max() 帽, 需实测.

扫描轴: g 梯度 / 阈值结构 (行中位 vs 全局中位 vs 分位数) / 双向缩放
(上半区 ×g, 下半区 ×s) / yh 符号一致性加权. 矩阵缓存 npz 二轮瞬时.
"""
import json
import os
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
SRC = ["gwps", "hct116_all"]
CACHE = "/data/resource/wanglab/aivc/aivc2026/tmp/bzr1/sim5_cache.npz"


def load_pairs(name, suffix, hold, col="logfc"):
    tbl = pq.read_table(f"{SIG}/{name}{suffix}.parquet",
                        columns=["target_gene", "gene", col],
                        filters=[("target_gene", "in", sorted(hold))])
    df = tbl.to_pandas()
    g = df.groupby(["target_gene", "gene"], sort=False)[col].mean().reset_index()
    g.columns = ["p", "g", col]
    return g


def build_matrices():
    hold = set(json.load(open(f"{ROOT}/data/interim/local_bench/Q_HepG2/targets.json")))
    hold_list = sorted(hold)
    truth = load_pairs("hepg2_all", "", hold, col="logfc").rename(columns={"logfc": "y"})
    parts = []
    for a in SRC:
        raw = load_pairs(a, "", hold).rename(columns={"logfc": "x"})
        cal = load_pairs(a, "_cal", hold).rename(columns={"logfc": "yh"})
        df = raw.merge(cal, on=["p", "g"], how="inner").merge(truth, on=["p", "g"], how="inner")
        parts.append(df[["p", "g", "x", "yh", "y"]])
    agg = pd.concat(parts, ignore_index=True).groupby(["p", "g"], sort=False)[
        ["x", "yh", "y"]].mean().reset_index()
    mx = agg.pivot(index="p", columns="g", values="x").reindex(hold_list)
    myh = agg.pivot(index="p", columns="g", values="yh").reindex(hold_list)
    mt = truth.pivot(index="p", columns="g", values="y").reindex(hold_list)
    ok = mx.notna().all(0) & myh.notna().all(0) & mt.notna().all(0)
    idx = pd.Index(ok.index)
    ok &= ~idx.isin(hold_list)
    ok &= (idx != "non-targeting")
    genes = ok.index[ok.to_numpy()]
    X = mx[genes].to_numpy(dtype=np.float32)
    YH = myh[genes].to_numpy(dtype=np.float32)
    Y = mt[genes].to_numpy(dtype=np.float32)
    np.savez_compressed(CACHE, X=X, YH=YH, Y=Y, hold=np.array(hold_list))
    return X, YH, Y, hold_list


def de_mask(Y, k=50):
    n, G = Y.shape
    k = min(k, G)
    ath = np.partition(np.abs(Y), -k, axis=1)[:, -k][:, None]
    return np.abs(Y) >= ath


def main():
    t0 = time.time()
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        X, YH, Y, hold_list = z["X"], z["YH"], z["Y"], list(z["hold"])
        print(f"[cache] {X.shape} loaded t={time.time()-t0:.0f}s", flush=True)
    else:
        X, YH, Y, hold_list = build_matrices()
        print(f"[built] {X.shape} t={time.time()-t0:.0f}s", flush=True)

    n = len(hold_list)
    mu = X.mean(0, keepdims=True)
    H = (X - mu).astype(np.float32)
    NC = np.where(X * H > 0, H, X).astype(np.float32)
    med_g = np.median(X, 0, keepdims=True)
    NCmed = np.where(X * (X - med_g) > 0, X - med_g, X).astype(np.float32)

    Rn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    DE = de_mask(Y)
    den = float(np.abs(Y)[DE].mean())
    Ysign = np.sign(Y)
    diag = np.arange(n)

    def evaluate(name, P, rows):
        Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        D = 1.0 - Pn @ Rn.T
        ranks = rankdata(D, axis=1)
        score = float(np.mean(1.0 - ranks[diag, diag] / n))
        agree = (np.sign(P) == Ysign) & (Ysign != 0)
        fp = float(np.mean((agree & DE).sum(1) / np.maximum(DE.sum(1), 1)))
        nmae_p = float(np.mean(np.abs(P - Y)[DE & np.isfinite(P)]) / den)
        rows.append((name, score, fp, nmae_p))
        return rows[-1]

    rows = []
    evaluate("A_x", X, rows)
    evaluate("H_center", H, rows)
    evaluate("NC", NC, rows)
    evaluate("NCmed", NCmed, rows)

    # --- 轴1: g 梯度 (行中位阈值, 单向放大) ---
    for g in (1.3, 1.6, 2.0, 2.5, 3.0, 4.0):
        thr = np.median(np.abs(NC), axis=1, keepdims=True)
        amp = np.where(np.abs(NC) > thr, g, 1.0).astype(np.float32)
        evaluate(f"AMP_r{g}", (NC * amp).astype(np.float32), rows)
    # --- 轴2: 全局中位阈值 ---
    gthr = np.median(np.abs(NC))
    for g in (1.6, 2.0):
        amp = np.where(np.abs(NC) > gthr, g, 1.0).astype(np.float32)
        evaluate(f"AMP_glob{g}", (NC * amp).astype(np.float32), rows)
    # --- 轴3: 分位数阈值 (75%) ---
    for g in (1.6, 2.0):
        thr = np.quantile(np.abs(NC), 0.75, axis=1, keepdims=True)
        amp = np.where(np.abs(NC) > thr, g, 1.0).astype(np.float32)
        evaluate(f"AMP_q75_{g}", (NC * amp).astype(np.float32), rows)
    # --- 轴4: 双向缩放 (上半区 ×g, 下半区 ×s) ---
    for g, s in ((1.6, 0.7), (2.0, 0.7), (1.6, 0.5), (2.0, 0.5)):
        thr = np.median(np.abs(NC), axis=1, keepdims=True)
        amp = np.where(np.abs(NC) > thr, g, s).astype(np.float32)
        evaluate(f"AMP2_{g}_{s}", (NC * amp).astype(np.float32), rows)
    # --- 轴5: yh 符号一致性加权 (双源共识放大) ---
    xh_ok = (np.sign(YH) == np.sign(NC)) & (np.abs(YH) > 1e-6)
    for g in (1.6, 2.0):
        amp = np.where(xh_ok, g, 1.0).astype(np.float32)
        evaluate(f"AMPcons_{g}", (NC * amp).astype(np.float32), rows)
    # --- 轴6: NCmed 基础 AMP ---
    for g in (1.6, 2.0):
        thr = np.median(np.abs(NCmed), axis=1, keepdims=True)
        amp = np.where(np.abs(NCmed) > thr, g, 1.0).astype(np.float32)
        evaluate(f"NCmedAMP{g}", (NCmed * amp).astype(np.float32), rows)

    print(f"\n== sim5b 全表 (t={time.time()-t0:.0f}s) ==", flush=True)
    A = rows[0]
    print(f"{'cand':16s} {'pds':>7s} {'dP':>8s} {'fidpx':>7s} {'dF':>8s} {'nmaepx':>7s} {'dN':>8s} gate", flush=True)
    best = None
    for name, p, f, nm in rows:
        dP, dF, dN = p - rows[0][1], f - rows[0][2], nm - rows[0][3]
        gate = dP >= 0.045 and dF >= -0.001 and dN <= 0.005
        flag = "PASS" if gate else ""
        if gate and (best is None or p > best[1]):
            best = (name, p, f, nm)
        print(f"{name:16s} {p:7.4f} {dP:+8.4f} {f:7.4f} {dF:+8.4f} {nm:7.4f} {dN:+8.4f} {flag}", flush=True)
    if best:
        print(f"\nBEST: {best[0]} pds={best[1]:.4f} (t={time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
