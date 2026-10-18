#!/usr/bin/env python
"""bzr1 修正9 — sim4 胜出构造 NC (sign-noncrossing 中心化) → 签名 parquet 落盘.

sim4 证据 (tmp/bzr1/hold_sim4b.log): NC_b1.0 pds d_vs_A=+0.0501 (≥HCX +0.0454) 且
fid 代理 dF=+0.0000 (与 v0.7 逐位同符号) — v0.8 官方 pds+0.0056/fid-0.0056 对冲的
解法: 中心化只发生在 sign(x) 不变的基因上, 翻转处退回 raw x.

用法: python tmp/bzr1/build_construct3.py NC
  NC:  v = x − μ_g;  若 x·v ≤ 0 (中心化翻转符号) 则 v = x
    μ_g = 每源全体 target_gene 按基因均值 (与 HCX 生产版同口径)

键集一致性沿用 cal_values assert 法则.
"""
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
SRC = ["gwps", "hct116_all"]


def load_x(name):
    """raw 聚合 (同 build_construct2.load_xy 的 raw 半边)."""
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
    del tbl
    return x, pt.dictionary, gt.dictionary, keys_u, NG


def main():
    tag = sys.argv[1]
    assert tag == "NC", f"未知构造 {tag}"
    for a in SRC:
        x, dp, dg, keys_u, NG = load_x(a)
        gidx = keys_u % NG
        cnt = np.bincount(gidx, minlength=NG).astype(np.float64)
        mu = (np.bincount(gidx, weights=x.astype(np.float64), minlength=NG) /
              np.maximum(cnt, 1.0))
        v = x - mu[gidx].astype(np.float32)
        flip = x * v <= 0
        nflip = int(flip.sum())
        v[flip] = x[flip]
        out = pd.DataFrame({
            "target_gene": dp.to_pandas().to_numpy()[keys_u // NG],
            "gene": dg.to_pandas().to_numpy()[keys_u % NG],
            "logfc": v.astype(np.float32),
        })
        path = f"{SIG}/{a}_cal{tag}.parquet"
        out.to_parquet(path)
        print(f"{a}[{tag}]: rows={len(out):,} flipped={nflip:,} ({nflip/len(out):.1%}) "
              f"max|v|={np.abs(v).max():.3f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
