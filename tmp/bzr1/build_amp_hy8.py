#!/usr/bin/env python
"""bzr1 iter037-续6 — amp_max_5src 集成: HY8 sel 门不变, 100 bench 靶 sel 位幅度替换 → 签名 parquet.

sim11 判决 (iter037-续5): amp_max_5src sim8c d8 +0.0037 首选 (max-form 只增不减, 风险下界=HY8).
TOPK=8 已溯源确证 (800 行 = 8/靶 × 100).

构造 (per-source 独立, build_construct5 模式逐行复刻, TOPK=8):
  基底  v0 = NC(x):  x − μ_g, 中心化翻符号处退回 raw x
  DE 集: 每 (源, target) 取 |yc| top-8 (de_mask 同式)
  sel  : de & (x*yc>0)                       — 与 calHY8 完全一致
  HY8  : sel 处 v = sign(x)·|yc|             — 基线 (须与既有 calHY8 文件逐键一致, assert)
  amp  : sel 处 v = sign(x)·max(|yc|, A_sq_5src[p,g])   — 仅 100 bench 靶
         A_sq_5src = amp_hold_v1.parquet 全 5 源按 (target,gene) 均值; NaN (无模型覆盖) → |yc|
         其余靶 (非 bench 100) sel 位幅度 = |yc|, 与 HY8 相同
输出: {SIG}/{a}_calHY8amp.parquet
"""
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
SRC = ["gwps", "hct116_all"]
TOPK = 8

t0 = time.time()

# ---- amp_hold_v1 → (target,gene) → A_sq (5src 均值) ----
amp = pd.read_parquet(f"{ROOT}/tmp/bzr1/amp_hold_v1.parquet")
agg = amp.groupby(["target_gene", "gene"], sort=False)["abs_sq"].mean()
amp_map = {}  # target_gene -> (gene_arr, a_sq_arr)
for tg, sub in amp.groupby("target_gene", sort=False):
    m = agg.loc[tg]
    amp_map[tg] = (m.index.to_numpy(), m.to_numpy(dtype=np.float32))
del amp, agg
print(f"amp_map targets={len(amp_map):,} t={time.time()-t0:.0f}s", flush=True)


def load_x(name):
    tbl = pq.read_table(f"{SIG}/{name}.parquet", columns=["target_gene", "gene", "logfc"])
    pt = pa.compute.dictionary_encode(tbl.column("target_gene")).combine_chunks()
    gt = pa.compute.dictionary_encode(tbl.column("gene")).combine_chunks()
    cp = np.asarray(pt.indices).astype(np.int64)
    cg = np.asarray(gt.indices).astype(np.int64)
    NG = len(gt.dictionary)
    keys = cp * NG + cg
    keys_u, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(keys_u)).astype(np.float64)
    x = (np.bincount(inv, weights=np.asarray(tbl.column("logfc"), dtype=np.float64),
                     minlength=len(keys_u)) / cnt).astype(np.float32)
    del tbl, keys, inv
    return x, pt.dictionary, gt.dictionary, keys_u, NG


def load_cal_y(name, dp, dg, NG):
    tblc = pq.read_table(f"{SIG}/{name}_cal.parquet", columns=["target_gene", "gene", "logfc"])
    ptc = pa.compute.dictionary_encode(tblc.column("target_gene")).combine_chunks()
    gtc = pa.compute.dictionary_encode(tblc.column("gene")).combine_chunks()
    mp = pd.Index(dp.to_pandas()).get_indexer(ptc.dictionary.to_pandas())
    mg = pd.Index(dg.to_pandas()).get_indexer(gtc.dictionary.to_pandas())
    assert (mp >= 0).all() and (mg >= 0).all(), f"{name}: cal 字典含 raw 之外键"
    kc = (mp[np.asarray(ptc.indices).astype(np.int64)] * NG
          + mg[np.asarray(gtc.indices).astype(np.int64)])
    yc = np.asarray(tblc.column("logfc"), dtype=np.float32)
    kcu, inv = np.unique(kc, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(kcu)).astype(np.float64)
    yc = (np.bincount(inv, weights=yc.astype(np.float64), minlength=len(kcu)) / cnt).astype(np.float32)
    del kc, inv, tblc, ptc, gtc, mp, mg
    return kcu, yc


def de_mask(keys_u, yc, NG):
    cp_u = (keys_u // NG).astype(np.int64)
    nT = int(cp_u.max()) + 1
    counts = np.bincount(cp_u, minlength=nT)
    starts = np.concatenate(([0], np.cumsum(counts)))[:-1]
    de = np.zeros(len(keys_u), dtype=bool)
    aby = np.abs(yc)
    for t in range(nT):
        s = int(starts[t]); L = int(counts[t])
        if L <= TOPK:
            de[s:s + L] = True
            continue
        loc = np.argpartition(aby[s:s + L], L - TOPK)[L - TOPK:]
        de[s + loc] = True
    return de, nT


def write_parquet(path, keys_u, dp, dg, NG, v):
    out = pd.DataFrame({
        "target_gene": dp.to_pandas().to_numpy()[keys_u // NG],
        "gene": dg.to_pandas().to_numpy()[keys_u % NG],
        "logfc": v.astype(np.float32),
    })
    out.to_parquet(path, index=False)
    del out


for a in SRC:
    x, dp, dg, keys_u, NG = load_x(a)
    print(f"[{a}] raw unique {len(x):,} t={time.time()-t0:.0f}s", flush=True)
    kcu, yc = load_cal_y(a, dp, dg, NG)
    assert len(kcu) == len(keys_u) and np.array_equal(kcu, keys_u), \
        f"{a}: cal/raw 键集不一致"
    assert not np.isnan(yc).any(), f"{a}: yc 含 NaN"
    del kcu
    # NC 基底
    gidx = keys_u % NG
    cnt = np.bincount(gidx, minlength=NG).astype(np.float64)
    mu = (np.bincount(gidx, weights=x.astype(np.float64), minlength=NG) /
          np.maximum(cnt, 1.0))
    v_nc = x - mu[gidx].astype(np.float32)
    flip = x * v_nc <= 0
    v_nc[flip] = x[flip]
    del mu, gidx, cnt
    de, nT = de_mask(keys_u, yc, NG)
    agree = x * yc > 0
    sel = de & agree
    aby = np.abs(yc)
    # HY8 基线 (逐键复刻, 供一致性 assert)
    v_hy8 = v_nc.copy()
    v_hy8[sel] = np.sign(x[sel]) * aby[sel]
    assert not np.isnan(v_hy8).any()
    # amp 集成: 仅 100 bench 靶的 sel 位幅度 = max(|yc|, A_sq_5src)
    tgt_names = dp.to_pandas().to_numpy()[keys_u // NG]
    is_bench = pd.Index(tgt_names).isin(amp_map)
    A = np.full(len(x), np.nan, dtype=np.float32)
    hit = 0
    for tg, (garr, aarr) in amp_map.items():
        rows = np.where(is_bench & (tgt_names == tg))[0]
        if len(rows) == 0:
            continue
        pos = pd.Index(garr).get_indexer(dg.to_pandas().to_numpy()[keys_u[rows] % NG])
        ok = pos >= 0
        A[rows[ok]] = aarr[pos[ok]]
        hit += int(ok.sum())
    n_bench_rows = int(is_bench.sum())
    print(f"[{a}] bench靶行={n_bench_rows:,} amp覆盖={hit:,} t={time.time()-t0:.0f}s", flush=True)
    v_amp = v_hy8.copy()
    m = sel & is_bench
    mag = np.where(np.isfinite(A[m]), np.maximum(aby[m], A[m]), aby[m])
    v_amp[m] = np.sign(x[m]) * mag
    assert not np.isnan(v_amp).any()
    n_mod = int(m.sum())
    chg = int((v_amp[m] != v_hy8[m]).sum())
    print(f"[{a}] sel&bench={n_mod:,} 幅度实改={chg:,} ({chg/max(n_mod,1):.1%}) "
          f"corr(x,amp)={np.corrcoef(x, v_amp)[0,1]:.3f} t={time.time()-t0:.0f}s", flush=True)
    write_parquet(f"{SIG}/{a}_calHY8amp.parquet", keys_u, dp, dg, NG, v_amp)
    # HY8 一致性诊断: 重建的 v_hy8 与既有 calHY8 文件应高度一致 (tie 级差异除外)
    tblv = pq.read_table(f"{SIG}/{a}_calHY8.parquet", columns=["logfc"])
    v_old = np.asarray(tblv.column("logfc"), dtype=np.float32)
    n_diff = int((v_old != v_hy8).sum())
    print(f"[{a}] HY8 重建 vs 既有文件: 不一致 {n_diff:,}/{len(x):,} ({n_diff/len(x):.3%}) "
          f"max|Δ|={np.abs(v_old - v_hy8).max():.4f} t={time.time()-t0:.0f}s", flush=True)
    del v_old, tblv
    del x, v_nc, yc, de, agree, sel, aby, keys_u, v_hy8, v_amp, A, tgt_names, is_bench
print(f"done t={time.time()-t0:.0f}s", flush=True)
