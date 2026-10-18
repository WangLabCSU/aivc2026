#!/usr/bin/env python
"""hold_sim v4 — fid 恢复臂筛选 (v0.8 官方裁决后: HCX pds+0.0056 被 fid-0.0056 精确对冲).

机制: 沿用 sim3 的 retrieval pds 模拟器 (bench pds_cosine 机制复刻), 新增 fid 代理
(per-pert truth-DE top-50 基因上的符号一致率) — 官方 v0.8 实测 fid 劣化的根源是
中心化翻转弱信号基因符号. 只用变换 pred 侧, real 侧固定.

候选家族 (都在 X 与 μ/σ 之间插值, 目标 = 保 HCX 检索增益同时护 fid):
  M_blend(β):   v = x − β·μ                      全局混合, β=1 即 HCX
  K_wiener(τ):  v = x − α_g·μ, α_g = sd²/(sd²+τ²)  按基因 Wiener 收缩 (sd 大→全中心化,
                sd 小→保 raw), τ 取 per-gene sd 分位数 → 尺度无关
  (HCX/JZX/A/B/C 复刻作锚点)

sanity 沿用 sim3: S1 B<A, S2 C<A (pds); 新增 F1: fid_proxy(HCX) < fid_proxy(A)
(官方已证 fid 劣化 → 模拟器 fid 代理须能复现此方向, 否则 fid 代理不可信).
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


def fid_proxy(P, Y, k=50):
    """per-pert fid 代理: truth |y| top-k DE 基因上 sign(pred)==sign(y) 占比."""
    n, G = Y.shape
    k = min(k, G)
    ath = np.partition(np.abs(Y), -k, axis=1)[:, -k][:, None]
    DE = np.abs(Y) >= ath
    agree = (np.sign(P) == np.sign(Y)) & (np.sign(Y) != 0)
    return (agree & DE).sum(1) / np.maximum(DE.sum(1), 1)


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

    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    a70 = 0.3 * X + 0.7 * YH
    # Wiener τ 网格: per-gene sd 分位数 (跨 100 pert 的 logfc 尺度)
    qs = np.percentile(sd, [25, 50, 75, 90])
    cands = {
        "A_x (v0.7)": X,
        "B_yhat (v0.8b)": YH,
        "H_center_x (v0.8)": (X - mu).astype(np.float32),
        "M_b0.3": (X - 0.3 * mu).astype(np.float32),
        "M_b0.5": (X - 0.5 * mu).astype(np.float32),
        "M_b0.7": (X - 0.7 * mu).astype(np.float32),
        "M_b0.9": (X - 0.9 * mu).astype(np.float32),
    }
    for q, tau in zip([25, 50, 75, 90], qs):
        al = (sd ** 2 / (sd ** 2 + tau ** 2)).astype(np.float32)
        cands[f"K_w{q}"] = (X - al * mu).astype(np.float32)
    # NC (non-crossing): 中心化但不许翻 sign(x) — fid 应精确保持 A 的符号模式
    for b in (0.5, 1.0):
        V = X - b * mu
        cands[f"NC_b{b}"] = np.where(X * V > 0, V, X).astype(np.float32)
    cands["C_a70"] = a70
    cands["H_center_a70"] = (a70 - a70.mean(0, keepdims=True)).astype(np.float32)

    Rn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    n = len(hold_list)

    print("\n== sim4: retrieval pds + fid 代理 (truth-DE top50 sign agreement) ==", flush=True)
    rows = []
    base_p = base_f = None
    for name, P in cands.items():
        Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        D = 1.0 - Pn @ Rn.T
        ranks = rankdata(D, axis=1)
        own = ranks[np.arange(n), np.arange(n)]
        score = 1.0 - own / n
        fp = float(np.mean(fid_proxy(P, Y)))
        if name.startswith("A"):
            base_p, base_f = float(np.mean(score)), fp
        rows.append((name, float(np.mean(score)), fp))
        print(f"{name:18s} pds={np.mean(score):.4f} d_vs_A={np.mean(score)-base_p:+.4f} "
              f"fidpx={fp:.4f} dF={fp-base_f:+.4f}", flush=True)

    print("\n== sanity ==", flush=True)
    d = {r[0]: r[1] for r in rows}
    f = {r[0]: r[2] for r in rows}
    s1 = d["B_yhat (v0.8b)"] < d["A_x (v0.7)"]
    s3 = abs(d["A_x (v0.7)"] - 0.5121) < 0.10
    f1 = f["H_center_x (v0.8)"] < f["A_x (v0.7)"]
    print(f"S1 B<A : {'PASS' if s1 else 'FAIL'}  S3 |A-0.5121|<0.10 : {'PASS' if s3 else 'FAIL'} "
          f"(A={d['A_x (v0.7)']:.4f})", flush=True)
    print(f"F1 fidpx(HCX)<fidpx(A) : {'PASS' if f1 else 'FAIL'} "
          f"(HCX={f['H_center_x (v0.8)']:.4f} A={f['A_x (v0.7)']:.4f}) — 官方 fid 劣化须被代理复现",
          flush=True)
    print(f"[done] t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
