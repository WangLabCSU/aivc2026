#!/usr/bin/env python
"""bzr1 v0.9b 符号门控混合 — sign(x)==sign(yhat) 保 raw 全幅; disagree 按 (1-β) 收缩.

机理: v0.8/α-blend 的 pds 崩因校准器把大基因质量压向先验 (corr(x,yhat)≈0.40, q99
0.126→0.048), 而 bench 靶的方向结构活在 raw 大基因里. 门控只在分歧基因上动刀:
若校准符号正确 → 减少错向质量 (fid/pds 双升); 若校准符号错 → 伤 pds. 纯经验题.

产出 β=0.5 / β=1.0 两套 ({src}_calSG50 / {src}_calSG100). β=0 即 v0.7.
"""
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
SRC = ["gwps", "hct116_all"]
BETAS = [(50, 0.5), (100, 1.0)]  # (文件后缀, β)


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


def main():
    t0 = time.time()
    for a in SRC:
        kr, xr, dp, dg, NG = load_sorted(a)
        tblc = pq.read_table(f"{SIG}/{a}_cal.parquet", columns=["target_gene", "gene", "logfc"])
        ptc = pa.compute.dictionary_encode(tblc.column("target_gene")).combine_chunks()
        gtc = pa.compute.dictionary_encode(tblc.column("gene")).combine_chunks()
        mp = pd.Index(dp.to_pandas()).get_indexer(ptc.dictionary.to_pandas())
        mg = pd.Index(dg.to_pandas()).get_indexer(gtc.dictionary.to_pandas())
        assert (mp >= 0).all() and (mg >= 0).all()
        kc = mp[np.asarray(ptc.indices).astype(np.int64)] * NG + mg[np.asarray(gtc.indices).astype(np.int64)]
        yc = np.asarray(tblc.column("logfc"), dtype=np.float32)
        kcu, inv = np.unique(kc, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(kcu)).astype(np.float64)
        yc = (np.bincount(inv, weights=yc.astype(np.float64), minlength=len(kcu)) / cnt).astype(np.float32)
        del kc, inv, tblc, ptc, gtc, mp, mg
        assert np.array_equal(kr, kcu), f"{a}: 键集不一致"
        cp = (kr // NG).astype(np.int64)
        cg = (kr % NG).astype(np.int64)
        gp = dp.to_pandas().to_numpy()
        gg = dg.to_pandas().to_numpy()
        disagree = (xr * yc) < 0  # yhat==0 → 积=0 → 不算分歧 (保守)
        nd = int(disagree.sum())
        frac = nd / len(xr)
        for tag, beta in BETAS:
            y = np.where(disagree, ((1 - beta) * xr).astype(np.float32), xr)
            out = pd.DataFrame({"target_gene": gp[cp], "gene": gg[cg], "logfc": y})
            out.to_parquet(f"{SIG}/{a}_calSG{tag}.parquet", index=False)
            del y, out
            print(f"[{a}] wrote calSG{tag} (β={beta}) t={time.time()-t0:.0f}s", flush=True)
        # 诊断: 分歧率 + 分歧基因上 |x| 的质量占比 (pds 风险敞口)
        expo = float(np.abs(xr[disagree]).sum() / np.abs(xr).sum())
        print(f"[{a}] disagree={nd:,} ({frac:.1%}) |x|质量敞口={expo:.3f} t={time.time()-t0:.0f}s", flush=True)
        del kr, xr, yc, cp, cg, disagree
    print(f"done t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
