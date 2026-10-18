#!/usr/bin/env python
"""bzr1 修正8 — hold_sim3 胜出构造 (跨pert对比归一) → 签名 parquet 落盘.

用法: python tmp/bzr1/build_construct2.py <TAG> <FORMULA>
  TAG:     构造名 (写 {src}_cal{TAG}.parquet)
  FORMULA: HCX | JZX | HCA70 | JZA70
    HCX:   按基因减去全 parquet target_gene 均值 (去共享响应) — sim3 H_center_x
    JZX:   再除以按基因 std (ddof=0, floor 1e-6)                — sim3 J_zscore_x
    HCA70: 对 a70=0.3x+0.7yh 做 HCX                            — sim3 H_center_a70
    JZA70: 对 a70 做 JZX                                       — sim3 J_zscore_a70

与 sim3 的差异 (bench 链裁决): sim3 在跨源聚合后的 100-hold-pert 矩阵上按轴0中心化;
这里在【每源聚合后全体 target_gene】上中心化 (生产版, 均值估计更稳, 近似不含自身).
变换后落盘, make_predictions 的双源均值/权重路径不变.

键集一致性沿用 cal_values assert 法则 (修正3/NG-per-source 律).
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
    """raw 与 cal 同键对齐, 返回 (x, y, 字典, keys_u, NG) 同序数组."""
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


def contrast(v, gidx, n_gene, mode):
    """按基因 (gidx) 跨 target_gene 中心化 / z-score (ddof=0, floor 1e-6)."""
    cnt = np.bincount(gidx, minlength=n_gene).astype(np.float64)
    mu = np.bincount(gidx, weights=v.astype(np.float64), minlength=n_gene) / np.maximum(cnt, 1.0)
    vc = v - mu[gidx].astype(np.float32)
    if mode == "center":
        return vc
    var = np.bincount(gidx, weights=(vc.astype(np.float64)) ** 2, minlength=n_gene) / np.maximum(cnt, 1.0)
    sd = np.sqrt(var)
    print(f"    std: min={sd.min():.4g} p1={np.percentile(sd, 1):.4g} med={np.median(sd):.4g} "
          f"n_gene={n_gene}", flush=True)
    return (vc / np.maximum(sd[gidx], 1e-6)).astype(np.float32)


def main():
    tag, form = sys.argv[1], sys.argv[2]
    assert form in ("HCX", "JZX", "HCA70", "JZA70"), f"未知构造 {form}"
    for a in SRC:
        x, y, dp, dg, keys_u, NG = load_xy(a)
        gidx = keys_u % NG          # 基因维索引 (0..NG-1)
        base = x if form in ("HCX", "JZX") else (0.3 * x + 0.7 * y)
        mode = "center" if form in ("HCX", "HCA70") else "zscore"
        v = contrast(base, gidx, NG, mode).astype(np.float32)
        out = pd.DataFrame({
            "target_gene": dp.to_pandas().to_numpy()[keys_u // NG],
            "gene": dg.to_pandas().to_numpy()[keys_u % NG],
            "logfc": v,
        })
        path = f"{SIG}/{a}_cal{tag}.parquet"
        out.to_parquet(path)
        print(f"{a}[{form}]: rows={len(out):,} max|v|={np.abs(v).max():.3f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
