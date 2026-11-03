#!/usr/bin/env python
"""hold_sim v3 — retrieval-style 模拟器, 复刻 bench pds_cosine 机制 (cell_eval2).

机制 (discrimination.py docstring 确认): 对每个 pert, 其 pred effect 与全部 n 个 real
effects 算 cosine 距离, own-pert 的升序排名 rank → score = 1 − rank/D (D=n, midrank 平局).
这是跨 pert 检索指标, 不是 per-pert 绝对对齐 — 解释 hold_sim2 矛盾 (ŷ 绝对对齐更好但
跨源可分性被破坏 → bench v0.8 崩).

sanity 预注册 (三条全过 → 模拟器可信):
  S1: B_yhat 检索分 须劣于 A_x   (bench v0.8 崩)
  S2: C_a70  检索分 须劣于 A_x   (bench a70 FAIL)
  S3: sim(A_x) ≈ 0.512 ± 0.10   (v0.7 官方 pds=0.5121)

新候选家族 H/J (跨pert对比归一, 只用自身预测, 不碰真值):
  H_center_x/yhat/a70: 按基因减去 100 pert 均值 (去共享响应, 保 pert 特异分量)
  J_zscore_x:          按基因 z-score (更强)
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


def main():
    t0 = time.time()
    hold = set(json.load(open(f"{ROOT}/data/interim/local_bench/Q_HepG2/targets.json")))
    hold_list = sorted(hold)
    print(f"hold perts={len(hold_list)}", flush=True)

    truth = load_pairs("hepg2_all", "", hold, col="logfc").rename(columns={"logfc": "y"})
    print(f"truth rows: {len(truth):,} t={time.time()-t0:.0f}s", flush=True)

    parts = []
    for a in SRC:
        raw = load_pairs(a, "", hold).rename(columns={"logfc": "x"})
        cal = load_pairs(a, "_cal", hold).rename(columns={"logfc": "yh"})
        df = raw.merge(cal, on=["p", "g"], how="inner").merge(truth, on=["p", "g"], how="inner")
        print(f"{a}: paired+truth={len(df):,} t={time.time()-t0:.0f}s", flush=True)
        parts.append(df[["p", "g", "x", "yh", "y"]])
    agg = pd.concat(parts, ignore_index=True).groupby(["p", "g"], sort=False)[
        ["x", "yh", "y"]].mean().reset_index()
    print(f"unique (p,g) pairs: {len(agg):,} t={time.time()-t0:.0f}s", flush=True)

    # ---- pert × gene 矩阵; 基因取全 pert 交集 (bench 中 n^2 距离阵共享同一基因集) ----
    mx = agg.pivot(index="p", columns="g", values="x").reindex(hold_list)
    myh = agg.pivot(index="p", columns="g", values="yh").reindex(hold_list)
    mt = truth.pivot(index="p", columns="g", values="y").reindex(hold_list)
    ok = mx.notna().all(0) & myh.notna().all(0) & mt.notna().all(0)
    # exclusion_scope="panel": 全部 panel 靶基因列一次性移除 (从 reference 的 perturbations 解析)
    ok &= ~pd.Index(ok.index).isin(hold_list)
    ok &= (pd.Index(ok.index) != "non-targeting")
    genes = ok.index[ok.to_numpy()]
    X = mx[genes].to_numpy(dtype=np.float32)
    YH = myh[genes].to_numpy(dtype=np.float32)
    Y = mt[genes].to_numpy(dtype=np.float32)
    print(f"matrix: {X.shape[0]} perts x {X.shape[1]} genes (all-pert 交集, 靶基因已剔除) "
          f"t={time.time()-t0:.0f}s", flush=True)

    # ---- 候选 (仅变换 pred 侧; real 侧固定) ----
    a70 = 0.3 * X + 0.7 * YH
    cands = {
        "A_x (v0.7)": X,
        "B_yhat (v0.8)": YH,
        "C_a70": a70,
        "D_signx_absyhat": (np.sign(X) * np.abs(YH)).astype(np.float32),
        "G_agree_recal": np.where((X * YH) >= 0, np.sign(X) * np.abs(YH), X).astype(np.float32),
        "H_center_x": (X - X.mean(0, keepdims=True)).astype(np.float32),
        "H_center_yhat": (YH - YH.mean(0, keepdims=True)).astype(np.float32),
        "H_center_a70": (a70 - a70.mean(0, keepdims=True)).astype(np.float32),
        "J_zscore_x": ((X - X.mean(0, keepdims=True)) /
                       np.maximum(X.std(0, keepdims=True), 1e-6)).astype(np.float32),
        "J_zscore_a70": ((a70 - a70.mean(0, keepdims=True)) /
                         np.maximum(a70.std(0, keepdims=True), 1e-6)).astype(np.float32),
    }

    # real effects 归一 (固定); cosine 距离阵 100x100
    Rn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    n = len(hold_list)

    print("\n== Q3 retrieval pds 模拟 (score=1-rank/n, midrank, cosine, panel-exclusion) ==", flush=True)
    base = None
    rows = []
    for name, P in cands.items():
        Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        D = 1.0 - Pn @ Rn.T
        ranks = rankdata(D, axis=1)  # 升序平均秩 = midrank
        own = ranks[np.arange(n), np.arange(n)]
        score = 1.0 - own / n
        top1 = float(np.mean(np.argmin(D, axis=1) == np.arange(n)))
        if name.startswith("A"):
            base = score
        gain = score - base
        rows.append((name, float(np.mean(score)), float(np.median(score)), top1,
                     float(np.mean(gain)), float(np.mean(gain > 0))))
        print(f"{name:18s} pds={np.mean(score):.4f} med={np.median(score):.4f} "
              f"top1={top1:.1%} d_vs_A={np.mean(gain):+.4f} win={np.mean(gain>0):.0%}", flush=True)

    print("\n== sanity ==", flush=True)
    d = {r[0]: r[1] for r in rows}
    s1 = d["B_yhat (v0.8)"] < d["A_x (v0.7)"]
    s2 = d["C_a70"] < d["A_x (v0.7)"]
    s3 = abs(d["A_x (v0.7)"] - 0.5121) < 0.10
    print(f"S1 B<A : {'PASS' if s1 else 'FAIL'}  S2 C<A : {'PASS' if s2 else 'FAIL'}  "
          f"S3 |A-0.5121|<0.10 : {'PASS' if s3 else 'FAIL'} (A={d['A_x (v0.7)']:.4f})", flush=True)
    print(f"[done] t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
