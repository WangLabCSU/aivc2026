#!/usr/bin/env python
"""bzr1 修正7 — hold_sim 胜出构造 → 签名 parquet 落盘 (gwps+hct116_all 双源).

用法: python tmp/bzr1/build_construct.py <TAG> <FORMULA>
  TAG:     构造名 (写 {src}_cal{TAG}.parquet)
  FORMULA: D | F70 | F50 | G  (与 hold_sim.py cands 字典一一对应)

输入: data/interim/signatures/{src}.parquet (x=raw) + {src}_cal.parquet (yh=校准)
键集一致性沿用 cal_values 的 assert 法则 (修正3/NG-per-source 律).
变换在【每源行级】做 (与 make_predictions 的双源均值路径一致), 不做 hold_sim 里的
跨源先聚合后变换 — 两者差异由 bench 链裁决.
"""
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
SRC = ["gwps", "hct116_all"]


def load_xy(name):
    """raw 与 cal 同键对齐, 返回 (x, y) 同序数组."""
    tbl = pq.read_table(f"{SIG}/{name}.parquet", columns=["target_gene", "gene", "logfc"])
    pt = pa.compute.dictionary_encode(tbl.column("target_gene")).combine_chunks()
    gt = pa.compute.dictionary_encode(tbl.column("gene")).combine_chunks()
    cp = np.asarray(pt.indices).astype(np.int64)
    cg = np.asarray(gt.indices).astype(np.int64)
    NG = len(gt.dictionary)
    keys = cp * NG + cg
    keys_u, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(keys_u)).astype(np.float64)

    def agg(v):
        return (np.bincount(inv, weights=v.astype(np.float64), minlength=len(keys_u)) / cnt).astype(np.float32)

    x = agg(np.asarray(tbl.column("logfc"), dtype=np.float32))
    del tbl

    tblc = pq.read_table(f"{SIG}/{name}_cal.parquet", columns=["target_gene", "gene", "logfc"])
    ptc = pa.compute.dictionary_encode(tblc.column("target_gene")).combine_chunks()
    gtc = pa.compute.dictionary_encode(tblc.column("gene")).combine_chunks()
    mp = pd.Index(pt.dictionary.to_pandas()).get_indexer(ptc.dictionary.to_pandas())
    mg = pd.Index(gt.dictionary.to_pandas()).get_indexer(gtc.dictionary.to_pandas())
    assert (mp >= 0).all() and (mg >= 0).all(), f"{name}: cal 字典外键"
    kc = mp[np.asarray(ptc.indices).astype(np.int64)] * NG + mg[np.asarray(gtc.indices).astype(np.int64)]
    kcu, inv2 = np.unique(kc, return_inverse=True)
    cnt2 = np.bincount(inv2, minlength=len(kcu)).astype(np.float64)
    y = (np.bincount(inv2, weights=np.asarray(tblc.column("logfc"), dtype=np.float64), minlength=len(kcu)) / cnt2).astype(np.float32)
    assert np.array_equal(keys_u, kcu), f"{name}: raw/cal 键集不一致"
    return x, y, pt.dictionary, gt.dictionary, keys_u, NG


def main():
    tag, form = sys.argv[1], sys.argv[2]
    assert form in ("D", "F70", "F50", "G"), f"未知构造 {form}"
    for a in SRC:
        x, y, dp, dg, keys_u, NG = load_xy(a)
        sx = np.sign(x)
        ax, ay = np.abs(x), np.abs(y)
        if form == "D":
            v = sx * ay
        elif form == "F70":
            v = sx * (0.7 * ax + 0.3 * ay)
        elif form == "F50":
            v = sx * (0.5 * ax + 0.5 * ay)
        else:  # G: 分歧保 raw
            v = np.where((x * y) >= 0, sx * ay, x)
        v = v.astype(np.float32)
        out = pd.DataFrame({
            "target_gene": dp.to_pandas().to_numpy()[keys_u // NG],
            "gene": dg.to_pandas().to_numpy()[keys_u % NG],
            "logfc": v,
        })
        path = f"{SIG}/{a}_cal{tag}.parquet"
        out.to_parquet(path)
        print(f"{a}: rows={len(out):,} -> {path} "
              f"(sign_flipped={(np.sign(v) != np.sign(x)).mean():.1%})", flush=True)


if __name__ == "__main__":
    main()
