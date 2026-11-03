#!/usr/bin/env python
"""bzr1 iter035-续6 — 幅度恢复臂 RS: 校准方向不变, 逐靶幅度重标定到 NC 范数.

依据 (probe_pds_decomp): YH 原始方向优于 NC (cosine 0.3588 vs 0.1572, spearman
0.2574 vs 0.0734), bench 崩因幅度收缩 (|YH|/|NC| med 0.7112) — λ·v 缩进 NTC 噪声
→ 采样均值被噪声淹没 → pds 崩. 修复 = 只恢复幅度, 方向保持 by-design:

  per target t:  s_t = ||NC_t||₂ / ||yc_t||₂,   v_t = s_t · yc_t

与 line-269 封禁的五臂不同: 那五臂全在「校准输出替换/混合原值」且保持收缩幅度或
改动方向; 本臂方向严格 = 校准器方向 (spearman 不变), 仅修 bench 崩的根因 (幅度).
零提交, 仅 bench. s_t 缺省 1 (||yc_t||=0 保护).

输入: data/interim/signatures/{src}.parquet (raw) + {src}_cal.parquet (yhat)
输出: data/interim/signatures/{src}_calRS.parquet (同键同序)
"""
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
SRC = ["gwps", "hct116_all"]


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


def main():
    t0 = time.time()
    for a in SRC:
        x, dp, dg, keys_u, NG = load_x(a)
        kcu, yc = load_cal_y(a, dp, dg, NG)
        assert np.array_equal(kcu, keys_u), f"{a}: cal/raw 键集不一致"
        del kcu
        # NC 基底 (同 build_construct5)
        gidx = keys_u % NG
        cnt = np.bincount(gidx, minlength=NG).astype(np.float64)
        mu = (np.bincount(gidx, weights=x.astype(np.float64), minlength=NG) /
              np.maximum(cnt, 1.0))
        v_nc = x - mu[gidx].astype(np.float32)
        flip = x * v_nc <= 0
        v_nc[flip] = x[flip]
        del mu, gidx, cnt
        # 逐靶范数比 s_t = ||NC_t|| / ||yc_t||
        cp_u = (keys_u // NG).astype(np.int64)
        nT = int(cp_u.max()) + 1
        n_nc = np.zeros(nT); n_yc = np.zeros(nT)
        np.add.at(n_nc, cp_u, (v_nc.astype(np.float64)) ** 2)
        np.add.at(n_yc, cp_u, (yc.astype(np.float64)) ** 2)
        s = np.sqrt(n_nc / np.maximum(n_yc, 1e-12))
        bad = ~np.isfinite(s) | (n_yc <= 0)
        s[bad] = 1.0
        v_rs = yc * s[cp_u].astype(np.float32)
        assert not np.isnan(v_rs).any()
        r_med = float(np.median(s[~bad])) if (~bad).any() else 1.0
        print(f"[{a}] targets={nT:,} s_med={r_med:.3f} s_q10={np.quantile(s[~bad],.1):.3f} "
              f"s_q90={np.quantile(s[~bad],.9):.3f} t={time.time()-t0:.0f}s", flush=True)
        out = pd.DataFrame({
            "target_gene": dp.to_pandas().to_numpy()[keys_u // NG],
            "gene": dg.to_pandas().to_numpy()[keys_u % NG],
            "logfc": v_rs.astype(np.float32),
        })
        out.to_parquet(f"{SIG}/{a}_calRS.parquet", index=False)
        del out, v_rs
        print(f"[{a}] corr(x,RS)={np.corrcoef(x, yc*s[cp_u])[0,1]:.3f} "
              f"q99|x|={np.quantile(np.abs(x),.99):.3f} "
              f"q99|RS|={np.quantile(np.abs(yc*s[cp_u]),.99):.3f} "
              f"max|RS|={np.abs(yc*s[cp_u]).max():.3f} t={time.time()-t0:.0f}s", flush=True)
        del x, v_nc, yc, keys_u, s
    print(f"done t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
