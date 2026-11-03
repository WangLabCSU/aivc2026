#!/usr/bin/env python
"""hold_sim v2 — 修复 v1 两大缺陷后的 hold 空间变换模拟.

v1 缺陷 (使 Q1/Q2 结论作废):
  1. rows_y 用源空间整数 key 查 hepg2 kmap — 跨源字典空间不同, 整数碰撞 → 真值错配.
  2. rows_g 用每源整数基因 ID 跨源 np.unique — 不同源同 ID 假性合并.
修复: 全程 (pert_name, gene_name) 字符串键 join; 读时 filters 下推只取 hold-pert 行.
sanity 预注册: C_a70 在 Q1b 不得优于 A_x (bench a70=FAIL); B_yhat 在 Q1 须明显劣于 A_x
(bench v0.8 崩). 两条不满足 → 模拟器仍无效, 结论不可信.
"""
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
SRC = ["gwps", "hct116_all"]


def load_pairs(name, suffix, hold, col="logfc"):
    """读 {name}{suffix}.parquet 的 hold-pert 行 → DataFrame[p, g, v] (均值聚合)."""
    tbl = pq.read_table(f"{SIG}/{name}{suffix}.parquet",
                        columns=["target_gene", "gene", col],
                        filters=[("target_gene", "in", sorted(hold))])
    df = tbl.to_pandas()
    g = df.groupby(["target_gene", "gene"], sort=False)[col].mean().reset_index()
    g.columns = ["p", "g", col]
    return g


def cosine_rows(A, B):
    num = (A * B).sum(1)
    den = np.sqrt((A * A).sum(1)) * np.sqrt((B * B).sum(1))
    return num / np.maximum(den, 1e-12)


def main():
    t0 = time.time()
    hold = set(json.load(open(f"{ROOT}/data/interim/local_bench/Q_HepG2/targets.json")))
    print(f"hold perts={len(hold)}", flush=True)

    # 真值: hepg2_all 限 hold perts
    truth = load_pairs("hepg2_all", "", hold, col="logfc").rename(columns={"logfc": "y"})
    print(f"truth rows: {len(truth):,} t={time.time()-t0:.0f}s", flush=True)

    parts = []
    for a in SRC:
        raw = load_pairs(a, "", hold).rename(columns={"logfc": "x"})
        cal = load_pairs(a, "_cal", hold).rename(columns={"logfc": "yh"})
        df = raw.merge(cal, on=["p", "g"], how="inner").merge(truth, on=["p", "g"], how="inner")
        print(f"{a}: raw={len(raw):,} paired+truth={len(df):,} t={time.time()-t0:.0f}s", flush=True)
        parts.append(df[["p", "g", "x", "yh", "y"]])
    alldf = pd.concat(parts, ignore_index=True)
    agg = alldf.groupby(["p", "g"], sort=False)[["x", "yh", "y"]].mean().reset_index()
    print(f"unique (pert,gene) pairs: {len(agg):,} perts={agg['p'].nunique()} t={time.time()-t0:.0f}s", flush=True)

    p_arr = agg["p"].to_numpy()
    x_u = agg["x"].to_numpy().astype(np.float32)
    yh_u = agg["yh"].to_numpy().astype(np.float32)
    y_u = agg["y"].to_numpy().astype(np.float32)
    perts = np.unique(p_arr)

    # ---- 候选变换 (与 v1 相同 cands) ----
    ax, ayh = np.abs(x_u), np.abs(yh_u)
    sx = np.sign(x_u)
    agree = (x_u * yh_u) >= 0
    cands = {
        "A_x (v0.7)": x_u,
        "B_yhat (v0.8)": yh_u,
        "C_a70": 0.3 * x_u + 0.7 * yh_u,
        "D_signx_absyhat": (sx * ayh).astype(np.float32),
        "F_w70_signx_mag": (sx * (0.7 * ax + 0.3 * ayh)).astype(np.float32),
        "F_w50_signx_mag": (sx * (0.5 * ax + 0.5 * ayh)).astype(np.float32),
        "G_agree_recal": np.where(agree, sx * ayh, x_u).astype(np.float32),
    }

    print("\n== Q1 全基因 per-pert cosine (vs y 真值) ==", flush=True)
    base = None
    for name, yh in cands.items():
        cs = np.full(len(perts), np.nan)
        for i, p in enumerate(perts):
            m = p_arr == p
            if m.sum() < 50:
                continue
            cs[i] = cosine_rows(yh[m][None, :], y_u[m][None, :])[0]
        if name.startswith("A"):
            base = cs
        gain = cs - base
        print(f"{name:22s} mean={np.nanmean(cs):+.4f} med={np.nanmedian(cs):+.4f} "
              f"win={np.nanmean(gain > 0):.1%} dmean={np.nanmean(gain):+.4f}", flush=True)

    print("\n== Q1b top-200|y| 限制 cosine ==", flush=True)
    base2 = None
    for name, yh in cands.items():
        cs = np.full(len(perts), np.nan)
        for i, p in enumerate(perts):
            m = p_arr == p
            if m.sum() < 400:
                continue
            top = np.argsort(-np.abs(y_u[m]))[:200]
            cs[i] = cosine_rows(yh[m][top][None, :], y_u[m][top][None, :])[0]
        if name.startswith("A"):
            base2 = cs
        gain = cs - base2
        print(f"{name:22s} mean={np.nanmean(cs):+.4f} med={np.nanmedian(cs):+.4f} "
              f"win={np.nanmean(gain > 0):.1%} dmean={np.nanmean(gain):+.4f}", flush=True)

    print("\n== Q2 per-pert NDCG@100 (relevance=|y|) 排序源对照 ==", flush=True)

    def ndcg(score):
        out = np.full(len(perts), np.nan)
        for i, p in enumerate(perts):
            m = p_arr == p
            if m.sum() < 400:
                continue
            order = np.argsort(-score[m])
            rel = np.abs(y_u[m])[order][:100]
            dcg = (2.0 ** rel - 1) / np.log2(np.arange(2, len(rel) + 2))
            ideal = np.sort(np.abs(y_u[m]))[::-1][:100]
            idcg = (2.0 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))
            out[i] = dcg.sum() / max(idcg.sum(), 1e-12)
        return out

    n_x = ndcg(ax)
    n_yh = ndcg(ayh)
    print(f"|x| (direct):       ndcg@100 mean={np.nanmean(n_x):.4f}", flush=True)
    print(f"|ŷ| (calibrator):   ndcg@100 mean={np.nanmean(n_yh):.4f}  "
          f"(>{np.nanmean(n_x):.4f} ? {'YES' if np.nanmean(n_yh) > np.nanmean(n_x) else 'NO'})", flush=True)

    acc_x = float(np.mean(np.sign(x_u) == np.sign(y_u)))
    acc_yh = float(np.mean(np.sign(yh_u) == np.sign(y_u)))
    top_mask = np.zeros(len(agg), dtype=bool)
    for i, p in enumerate(perts):
        m = p_arr == p
        if m.sum() < 400:
            continue
        top = np.argsort(-np.abs(y_u[m]))[:200]
        top_mask[np.where(m)[0][top]] = True
    acc_x_top = float(np.mean(np.sign(x_u[top_mask]) == np.sign(y_u[top_mask])))
    acc_yh_top = float(np.mean(np.sign(yh_u[top_mask]) == np.sign(y_u[top_mask])))
    print(f"\nsign acc 全体: x={acc_x:.3f} ŷ={acc_yh:.3f} | top200|y|: x={acc_x_top:.3f} ŷ={acc_yh_top:.3f}", flush=True)
    print(f"[done] t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
