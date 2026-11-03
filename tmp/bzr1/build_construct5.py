#!/usr/bin/env python
"""bzr1 iter035-续2 — 主线结构臂 NC+yh 混合 → 签名 parquet 落盘 (两档括号一次建成).

progress-doc 候选 (iter035-续): 「NC+yh 混合 (DE-gene yh 幅度 + NC sign lock,
须守 A70 教训: 大幅混入 cosine 主导签名崩 pds)」.

构造 (per-source 独立):
  基底  v0 = NC(x):  x − μ_g, 中心化翻符号处退回 raw x (build_construct3 同式)
  DE 集: 每 (源, target) 取 |yc| (校准器 yhat) top-25 — DE 定义来自校准器自身置信,
        不触碰 truth (Q_HepG2 truth 仅评分用, 构造禁入防泄漏).
  HY :  DE ∩ sign(x)==sign(yc) 处 v = sign(x)·|yc| (幅度让位校准器), 其余 = NC(x).
        cosine 结构仅 25 位/靶受扰, 幅度收缩限局部 (S08 教训: 均匀缩已败, 局部≠均匀).
  HYf:  同 DE 集, v = sign(x)·max(|x|,|yc|) (只升不缩的受控 AMP 括号对).

键集一致性沿用 assert 法则 (cal 键集 == raw 键集, blend_cal 同款 assert).
"""
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
SRC = ["gwps", "hct116_all"]
TOPK = int(sys.argv[1]) if len(sys.argv) > 1 else 25  # DE-gene 数/靶/源 — A70 教训: 少碰 cosine 主导结构
TAG = sys.argv[2] if len(sys.argv) > 2 else "HY"


def load_x(name):
    """raw 聚合 (同 build_construct3.load_x)."""
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
    """校准器 yhat → 与 raw keys_u 同序对齐 (blend_cal 字典映射同式)."""
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
    """每 target 取 |yc| top-TOPK 的行掩码 (keys_u 已按 (target,gene) 升序 → 组连续)."""
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


def main():
    t0 = time.time()
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
        print(f"[{a}] targets={nT:,} de={int(de.sum()):,} de&agree={int(sel.sum()):,} "
              f"({int(sel.sum())/len(x):.2%}) t={time.time()-t0:.0f}s", flush=True)
        # HY: DE∩sign 同向处幅度让位 yc
        v_hy = v_nc.copy()
        v_hy[sel] = np.sign(x[sel]) * aby[sel]
        assert not np.isnan(v_hy).any()
        corr_hy = np.corrcoef(x, v_hy)[0, 1]
        out = pd.DataFrame({
            "target_gene": dp.to_pandas().to_numpy()[keys_u // NG],
            "gene": dg.to_pandas().to_numpy()[keys_u % NG],
            "logfc": v_hy.astype(np.float32),
        })
        out.to_parquet(f"{SIG}/{a}_cal{TAG}.parquet", index=False)
        del out, v_hy
        # HYf: 同 DE 集, 只升不缩
        v_hyf = v_nc.copy()
        v_hyf[sel] = np.sign(x[sel]) * np.maximum(np.abs(x[sel]), aby[sel])
        assert not np.isnan(v_hyf).any()
        out = pd.DataFrame({
            "target_gene": dp.to_pandas().to_numpy()[keys_u // NG],
            "gene": dg.to_pandas().to_numpy()[keys_u % NG],
            "logfc": v_hyf.astype(np.float32),
        })
        out.to_parquet(f"{SIG}/{a}_cal{TAG}f.parquet", index=False)
        del out, v_hyf
        # 诊断: 与 raw 的逐键相关 (cosine 结构保持度) + 幅度分位
        print(f"[{a}] corr(x,NC)={np.corrcoef(x, v_nc)[0,1]:.3f} "
              f"corr(x,HY)={corr_hy:.3f} "
              f"q99|x|={np.quantile(np.abs(x), .99):.3f} "
              f"q99|yc|={np.quantile(aby, .99):.3f} "
              f"max|x|={np.abs(x).max():.3f} t={time.time()-t0:.0f}s", flush=True)
        del x, v_nc, yc, de, agree, sel, aby, keys_u
    print(f"done t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
