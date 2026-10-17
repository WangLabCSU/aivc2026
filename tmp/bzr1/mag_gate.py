#!/usr/bin/env python
"""bzr1 v0.9c 幅度门控混合 — |x|≥τ 保 raw (pds 载体豁免); 尾部换 yhat_cal.

证据链: v0.8 全域校准 pds 崩 (top-gene 重构); sg100 清零分歧基因 pds 反降
(分歧基因 32% |x| 质量为真信号) → 唯一未测角落 = 幅度分域: 大 |x| 基因
(方向结构载体) 原样保留, 小 |x| 尾部 (fid/nmae 噪声载体) 全换校准值.

产出 MG75 (|x|≥q75 保 raw) / MG90 (|x|≥q90 保 raw) 两套. 附诊断:
分歧 |x| 质量按 |x| 分位带分解 → 判定 pds 载体是否真被 τ 豁免.
"""
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
SRC = ["gwps", "hct116_all"]
GATES = [(75, 0.75), (90, 0.90)]  # (后缀, q)


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
        ax = np.abs(xr)
        # 诊断: |x| 分位带 × 分歧质量分布 → pds 载体是否落进被改写区
        disagree = (xr * yc) < 0
        qs = [0.5, 0.75, 0.9, 0.95]
        cuts = np.quantile(ax, qs)
        print(f"[{a}] |x| 分位 cuts q50/75/90/95 = {np.round(cuts,4)}", flush=True)
        for i, q in enumerate(qs):
            lo = 0.0 if i == 0 else cuts[i - 1]
            band = (ax > lo) & (ax <= cuts[i])
            m = band & disagree
            print(f"[{a}]  band(>{lo:.4f},{cuts[i]:.4f}] rows={band.sum():,} "
                  f"disagree={m.sum():,} |x|mass={ax[m].sum()/ax.sum():.3f}", flush=True)
        for tag, qv in GATES:
            tau = np.quantile(ax, qv)
            keep = ax >= tau
            y = np.where(keep, xr, yc).astype(np.float32)
            out = pd.DataFrame({"target_gene": gp[cp], "gene": gg[cg], "logfc": y})
            out.to_parquet(f"{SIG}/{a}_calMG{tag}.parquet", index=False)
            del y, out
            print(f"[{a}] wrote calMG{tag} (τ=q{qv}={tau:.4f}, 改写率={1-keep.mean():.1%}) t={time.time()-t0:.0f}s", flush=True)
        del kr, xr, yc, cp, cg, ax, disagree
    print(f"done t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
