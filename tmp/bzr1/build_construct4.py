#!/usr/bin/env python
"""bzr1 iter034 — sim5b 胜出构造 AMP (q75 阈值选择性放大, 保号) → 签名 parquet 落盘.

sim5b 证据 (tmp/bzr1/sim5b.log): AMP_q75_2.0 pds d_vs_A=+0.0545 (>NC +0.0501,
≥HCX +0.0454) 且 fid 代理 dF=+0.0000 + nmae 代理 dN=-0.0256 — 首支 pds/nmae/fid
三轴同时不劣于 v0.7 的候选. 机制: NC 中心化基础上, 对 |v|>行75分位 的符号自信
基因 ×2.0 放大 — 推深进 pred 显著集 (fid 的 n_pred 杠杆), 拉开 retrieval 余弦
对比度, 压 nmae.

用法: python tmp/bzr1/build_construct4.py AMPQ75
  AMPQ75: v = NC(x);  thr = 行75分位(|v|);  v[|v|>thr] *= 2.0
    NC = sign-noncrossing 中心化 (build_construct3 口径, μ_g 每源全体均值)
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
    """raw 聚合 (同 build_construct3.load_x, 逐位同口径)."""
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
    assert tag == "AMPQ75", f"未知构造 {tag}"
    for a in SRC:
        x, dp, dg, keys_u, NG = load_x(a)
        gidx = keys_u % NG
        cnt = np.bincount(gidx, minlength=NG).astype(np.float64)
        mu = (np.bincount(gidx, weights=x.astype(np.float64), minlength=NG) /
              np.maximum(cnt, 1.0))
        v = x - mu[gidx].astype(np.float32)
        flip = x * v <= 0
        v[flip] = x[flip]
        # AMP: 行75分位阈值选择性放大 (逐 target_gene 行, 按 load_x 聚合行序)
        rows_per_p = np.bincount(keys_u // NG, minlength=len(dp)).astype(np.int64)
        # 逐 target_gene 分组计算 75 分位 — 用 pandas 按行块切分最稳妥
        p_ids = keys_u // NG
        v2 = v.copy()
        row_starts = np.searchsorted(p_ids, np.arange(len(dp)))
        row_ends = np.append(row_starts[1:], len(v))
        thr = np.empty(len(dp), dtype=np.float32)
        for i in range(len(dp)):
            seg = np.abs(v[row_starts[i]:row_ends[i]])
            if len(seg):
                thr[i] = np.quantile(seg, 0.75)
            else:
                thr[i] = 0.0
        amp = np.where(np.abs(v) > thr[p_ids], 2.0, 1.0).astype(np.float32)
        v2 = (v * amp).astype(np.float32)
        out = pd.DataFrame({
            "target_gene": dp.to_pandas().to_numpy()[keys_u // NG],
            "gene": dg.to_pandas().to_numpy()[keys_u % NG],
            "logfc": v2,
        })
        path = f"{SIG}/{a}_cal{tag}.parquet"
        out.to_parquet(path)
        namp = int((amp > 1).sum())
        print(f"{a}[{tag}]: rows={len(out):,} amplified={namp:,} ({namp/len(out):.1%}) "
              f"max|v|={np.abs(v2).max():.3f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
