#!/usr/bin/env python
"""bzr1 修正7 前置 — hold 空间变换模拟: 任何 signature 变换先对真值算 per-pert cosine, 不烧 bench.

证据链: 手术家族五点全败 (修正6), 但均为"整段替换"构造. 本模拟在 bench-100 perts 的
hepg2 真值上逐 pert 评估候选变换 T(x, ŷ) → ŷ', 回答两问:
  Q1 (构造扫描): sign(x)·|ŷ| 等方向保持构造是否优于 x (v0.7)?  — 优于才值得建 parquet+bench
  Q2 (重训练 headroom): |ŷ| 的 per-pert NDCG@100 是否优于 |x|?  — 决定 rank 目标重训练臂是否开工

对照: a70 (α=0.7 blend) 应复现劣化 → 模拟与 bench 方向一致的 sanity check.
输出: 每构造 per-pert cosine (full + top200|y| 限制) 的 mean/median/胜率; NDCG 对照表.
"""
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
SRC = ["gwps", "hct116_all"]
HOLD_PERTS = set(json.load(open(f"{ROOT}/data/interim/local_bench/Q_HepG2/targets.json"))) \
    if False else None  # 占位, 下面真加载


def load_sorted(name):
    tbl = pq.read_table(f"{SIG}/{name}.parquet", columns=["target_gene", "gene", "logfc"])
    pt = pa.compute.dictionary_encode(tbl.column("target_gene")).combine_chunks()
    gt = pa.compute.dictionary_encode(tbl.column("gene")).combine_chunks()
    cp = np.asarray(pt.indices).astype(np.int64)
    cg = np.asarray(gt.indices).astype(np.int64)
    NG = len(gt.dictionary)
    keys = cp * NG + cg
    lf = np.asarray(tbl.column("logfc"), dtype=np.float32)
    del tbl
    keys_u, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(keys_u)).astype(np.float64)
    ssum = np.bincount(inv, weights=lf.astype(np.float64), minlength=len(keys_u))
    lf = (ssum / cnt).astype(np.float32)
    del keys, inv
    return keys_u, lf, pt.dictionary, gt.dictionary, NG


def cal_values(name, kr, dp, dg, NG):
    tbl = pq.read_table(f"{SIG}/{name}_cal.parquet", columns=["target_gene", "gene", "logfc"])
    ptc = pa.compute.dictionary_encode(tbl.column("target_gene")).combine_chunks()
    gtc = pa.compute.dictionary_encode(tbl.column("gene")).combine_chunks()
    mp = pd.Index(dp.to_pandas()).get_indexer(ptc.dictionary.to_pandas())
    mg = pd.Index(dg.to_pandas()).get_indexer(gtc.dictionary.to_pandas())
    kc = mp[np.asarray(ptc.indices).astype(np.int64)] * NG + mg[np.asarray(gtc.indices).astype(np.int64)]
    yc = np.asarray(tbl.column("logfc"), dtype=np.float32)
    kcu, inv = np.unique(kc, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(kcu)).astype(np.float64)
    yc = (np.bincount(inv, weights=yc.astype(np.float64), minlength=len(kcu)) / cnt).astype(np.float32)
    assert np.array_equal(kr, kcu), f"{name}: 键集不一致"
    return yc


def cosine_rows(A, B):
    """行向 cosine, A/B 同形 (n_pert × n_gene)."""
    num = (A * B).sum(1)
    den = np.sqrt((A * A).sum(1)) * np.sqrt((B * B).sum(1))
    return num / np.maximum(den, 1e-12)


def main():
    import json
    t0 = time.time()
    hold = set(json.load(open(f"{ROOT}/data/interim/local_bench/Q_HepG2/targets.json")))
    print(f"hold perts={len(hold)}", flush=True)

    ky, y, dyp, dyg, NG = load_sorted("hepg2_all")
    py = dyp.to_pandas().to_numpy()[ky // NG]
    mask_h = np.isin(py, list(hold))
    # 只保留 bench-100 perts 的真值行
    kmap = {}
    for k, v in zip(ky[mask_h], y[mask_h]):
        kmap[k] = v
    print(f"hepg2 truth rows on hold perts: {mask_h.sum():,} t={time.time()-t0:.0f}s", flush=True)

    # 逐源收集 (pert, gene) 对齐行 → 聚合到 pert × gene 矩阵 (两源取平均)
    rows_p, rows_g, rows_x, rows_yh, rows_y = [], [], [], [], []
    for a in SRC:
        kr, xr, dp, dg, NG = load_sorted(a)  # 注意: NG 随源变 (hepg2=9622, gwps=8248), 不能跨源复用
        yc = cal_values(a, kr, dp, dg, NG)
        cp = dp.to_pandas().to_numpy()[kr // NG]
        m = np.isin(cp, list(hold))
        have = np.array([k in kmap for k in kr[m]])
        idx = np.where(m)[0][have]
        rows_p.append(cp[idx]); rows_g.append((kr[idx] % NG))
        rows_x.append(xr[idx]); rows_yh.append(yc[idx]); rows_y.append(
            np.array([kmap[k] for k in kr[idx]], dtype=np.float32))
        del kr, xr, yc, cp, m, have, idx
        print(f"{a}: hold-paired rows={len(rows_x[-1]):,} t={time.time()-t0:.0f}s", flush=True)
    p_all = np.concatenate(rows_p); g_all = np.concatenate(rows_g)
    x_all = np.concatenate(rows_x); yh_all = np.concatenate(rows_yh); y_all = np.concatenate(rows_y)
    # 两源同 (pert,gene) 取均值 — 基因名字符串统一编号空间 (gwps/hct116 字典词序不同)
    pd_u, p_inv = np.unique(p_all, return_inverse=True)
    gd_u, g_inv = np.unique(g_all, return_inverse=True)
    NGu = len(gd_u)
    kg = p_inv.astype(np.int64) * NGu + g_inv
    kg_u, inv = np.unique(kg, return_inverse=True)
    w = np.bincount(inv).astype(np.float64)
    def avg(v):
        return (np.bincount(inv, weights=v.astype(np.float64), minlength=len(kg_u)) / w).astype(np.float32)
    x_u, yh_u, y_u = avg(x_all), avg(yh_all), avg(y_all)
    p_u = pd_u[kg_u // NGu]
    del rows_p, rows_g, rows_x, rows_yh, rows_y, x_all, yh_all, y_all
    print(f"unique pert-gene pairs on hold: {len(kg_u):,} t={time.time()-t0:.0f}s", flush=True)

    perts = np.unique(p_u)
    print(f"perts covered: {len(perts)}", flush=True)

    # ---- 候选变换 ----
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
        "G_agree_recal": np.where(agree, sx * ayh, x_u).astype(np.float32),  # 分歧基因完全保 raw
    }

    # 全基因 per-pert cosine
    print("\n== Q1 全基因 per-pert cosine (vs y 真值) ==", flush=True)
    base = None
    for name, yh in cands.items():
        cs = np.full(len(perts), np.nan)
        for i, p in enumerate(perts):
            m = p_u == p
            if m.sum() < 50:
                continue
            cs[i] = cosine_rows(yh[m][None, :], y_u[m][None, :])[0]
        if name.startswith("A"):
            base = cs
        gain = cs - base
        print(f"{name:22s} mean={np.nanmean(cs):+.4f} med={np.nanmedian(cs):+.4f} "
              f"win={np.nanmean(gain > 0):.1%} dmean={np.nanmean(gain):+.4f}", flush=True)

    # top-200|y| 限制 (近似 DE 选择域)
    print("\n== Q1b top-200|y| 限制 cosine ==", flush=True)
    base2 = None
    for name, yh in cands.items():
        cs = np.full(len(perts), np.nan)
        for i, p in enumerate(perts):
            m = p_u == p
            if m.sum() < 400:
                continue
            top = np.argsort(-np.abs(y_u[m]))[:200]
            cs[i] = cosine_rows(yh[m][top][None, :], y_u[m][top][None, :])[0]
        if name.startswith("A"):
            base2 = cs
        gain = cs - base2
        print(f"{name:22s} mean={np.nanmean(cs):+.4f} med={np.nanmedian(cs):+.4f} "
              f"win={np.nanmean(gain > 0):.1%} dmean={np.nanmean(gain):+.4f}", flush=True)

    # ---- Q2: 排序 headroom (NDCG@100, relevance=|y|) ----
    print("\n== Q2 per-pert NDCG@100 (relevance=|y|) 排序源对照 ==", flush=True)

    def ndcg(score):
        out = np.full(len(perts), np.nan)
        for i, p in enumerate(perts):
            m = p_u == p
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
    n_oracle_sign = ndcg(np.where(agree, ayh, ax))  # 分歧取 |x| 一致取 |ŷ| 的混排
    print(f"|x| (direct):        ndcg@100 mean={np.nanmean(n_x):.4f}", flush=True)
    print(f"|ŷ| (calibrator):    ndcg@100 mean={np.nanmean(n_yh):.4f}  (>{np.nanmean(n_x):.4f} ? {'YES' if np.nanmean(n_yh) > np.nanmean(n_x) else 'NO'})", flush=True)
    print(f"|x|∪|ŷ| 分域混排:   ndcg@100 mean={np.nanmean(n_oracle_sign):.4f}", flush=True)

    # 符号准确率: x vs ŷ, 全体 + top200|y|
    acc_x = float(np.mean(np.sign(x_u) == np.sign(y_u)))
    acc_yh = float(np.mean(np.sign(yh_u) == np.sign(y_u)))
    top_mask = np.zeros(len(kg_u), dtype=bool)
    for i, p in enumerate(perts):
        m = p_u == p
        if m.sum() < 400:
            continue
        top = np.argsort(-np.abs(y_u[m]))[:200]
        ii = np.where(m)[0][top]
        top_mask[ii] = True
    acc_x_top = float(np.mean(np.sign(x_u[top_mask]) == np.sign(y_u[top_mask])))
    acc_yh_top = float(np.mean(np.sign(yh_u[top_mask]) == np.sign(y_u[top_mask])))
    print(f"\nsign acc 全体: x={acc_x:.3f} ŷ={acc_yh:.3f} | top200|y|: x={acc_x_top:.3f} ŷ={acc_yh_top:.3f}", flush=True)
    print(f"[done] t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
