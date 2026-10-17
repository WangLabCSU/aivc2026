#!/usr/bin/env python
"""bzr1 v0.9 混合臂: 签名级 α-blend — y = α·x_raw + (1-α)·yhat_cal (iter030-修正5).

动机: v0.8 纯校准签名 bench 5/6 raw 改善但 pds 崩 (0.8902→0.5520, cosine 尺度不变
→ 真方向退化); 校准器 yhat 大幅收缩 (gwps q99 0.126→0.048) 重排了 top-gene 结构.
混合保留 raw 主方向同时掺入校准幅度修正. α=1 等价 v0.7, α=0 等价 v0.8.

对齐: cal.parquet 行序 = raw 唯一键升序 (make_delta_v2 落盘序), raw 侧用同法
np.unique+bincount-mean 重排; 每源独立字典码空间, cal 字典经字符串映射回 raw 空间.

一次构建同时产出 α=0.5 / α=0.7 两套 parquet ({src}_calA50.parquet / {src}_calA70.parquet).
"""
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
SRC = ["gwps", "hct116_all"]
ALPHAS = [(50, 0.5), (70, 0.7)]  # (文件后缀, α)


def load_sorted(name):
    """读 parquet → (unique sorted keys, 值 bincount-mean, 字典) — 与 make_delta_v2 同式."""
    tbl = pq.read_table(f"{SIG}/{name}.parquet", columns=["target_gene", "gene", "logfc"])
    pt = pa.compute.dictionary_encode(tbl.column("target_gene"))
    gt = pa.compute.dictionary_encode(tbl.column("gene"))
    pt, gt = pt.combine_chunks(), gt.combine_chunks()
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


def main():
    t0 = time.time()
    for a in SRC:
        kr, xr, dp, dg, NG = load_sorted(a)
        print(f"[{a}] raw unique {len(kr):,} t={time.time()-t0:.0f}s", flush=True)
        # cal 字典码 → raw 字典码 (字符串映射, 字典仅数千级)
        tblc = pq.read_table(f"{SIG}/{a}_cal.parquet", columns=["target_gene", "gene", "logfc"])
        ptc = pa.compute.dictionary_encode(tblc.column("target_gene")).combine_chunks()
        gtc = pa.compute.dictionary_encode(tblc.column("gene")).combine_chunks()
        mp = pd.Index(dp.to_pandas()).get_indexer(ptc.dictionary.to_pandas())
        mg = pd.Index(dg.to_pandas()).get_indexer(gtc.dictionary.to_pandas())
        assert (mp >= 0).all() and (mg >= 0).all(), f"{a}: cal 字典含 raw 之外键"
        kc = mp[np.asarray(ptc.indices).astype(np.int64)] * NG + mg[np.asarray(gtc.indices).astype(np.int64)]
        yc = np.asarray(tblc.column("logfc"), dtype=np.float32)
        kcu, inv = np.unique(kc, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(kcu)).astype(np.float64)
        yc = (np.bincount(inv, weights=yc.astype(np.float64), minlength=len(kcu)) / cnt).astype(np.float32)
        del kc, inv, tblc, ptc, gtc, mp, mg
        assert len(kcu) == len(kr), f"{a}: raw {len(kr)} vs cal {len(kcu)} 键数不一致"
        assert np.array_equal(kr, kcu), f"{a}: 键集不一致"
        cp = (kr // NG).astype(np.int64)
        cg = (kr % NG).astype(np.int64)
        gp = dp.to_pandas().to_numpy()
        gg = dg.to_pandas().to_numpy()
        for tag, alpha in ALPHAS:
            y = (alpha * xr + (1 - alpha) * yc).astype(np.float32)
            out = pd.DataFrame({"target_gene": gp[cp], "gene": gg[cg], "logfc": y})
            out.to_parquet(f"{SIG}/{a}_calA{tag}.parquet", index=False)
            del y, out
            print(f"[{a}] wrote calA{tag} t={time.time()-t0:.0f}s", flush=True)
        # 落盘自检: 与 raw/cal 的逐键相关 (诊断混合几何)
        print(f"[{a}] corr(x,ycal)={np.corrcoef(xr, yc)[0,1]:.3f} "
              f"corr(x,blend50)={np.corrcoef(xr, 0.5*xr+0.5*yc)[0,1]:.3f} t={time.time()-t0:.0f}s", flush=True)
        del kr, xr, yc, cp, cg
    print(f"done t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
