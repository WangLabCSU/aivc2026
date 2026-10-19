#!/usr/bin/env python
"""bzr1 v3 生产构建 — V3(K=4, β=0.5) direct-y 校准输出 + 行空间部分去相关.

预注册来源: tmp/bzr1/probe_v4_meta.json — 全 NG 宽度 (K,β) 网格唯一 two-win 点
K4_b0.5 (pds 0.8994 > max(NC 0.7935, rawcal 0.5545), sp 0.0917 ≥ sp_nc−0.02).
构造与 train_v3_probe.fit_basis/transform 逐式一致:

  u_1 = 行均值方向, u_{2..K+1} = 去均值行矩阵的 top-K 右奇异向量
  P' = P − β · Σ_{k≤K+1} (P·u_k) u_k

每源独立: yc (校准 yhat, 键序=raw 唯一键升序) 按全局 6-CTX 基因词汇表展开成
per-target 行矩阵 (缺键补 0, probe 约定), basis 只从本源预测行自身几何拟合
(无真值泄漏), 变换后取回本源键位写 {src}_calV3.parquet (同键同序, 与
build_rs.py 落盘约定一致).

三重代理迁移失败律: probe 读数不可信, bench 链才是唯一仲裁. 零提交, 仅 bench.

输入: data/interim/signatures/{src}.parquet (raw) + {src}_cal.parquet (yhat)
输出: data/interim/signatures/{src}_calV3.parquet
"""
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
SRC = ["gwps", "hct116_all"]
CTX = ["gwps", "hct116_all", "hepg2_all", "jurkat_all", "rpe1_all", "k562_ess_all"]
K, BETA = 4, 0.5
NG_EXPECT = 39084  # probe 实测全局 6-CTX 基因词汇宽度; 变了则 probe 迁移前提破裂


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
    genes = set()
    for n in CTX:
        df = pd.read_parquet(f"{SIG}/{n}.parquet", columns=["gene"])
        genes |= set(pd.unique(df["gene"].astype(str)))
        del df
    vocab_g = pd.Index(sorted(genes))
    NG = len(vocab_g)
    del genes
    print(f"global vocab g={NG} (expect {NG_EXPECT}) t={time.time()-t0:.0f}s", flush=True)
    assert NG == NG_EXPECT, f"全局词汇 {NG} != probe 实测 {NG_EXPECT}, probe 迁移前提破坏"

    for a in SRC:
        x, dp, dg, keys_u, ngs = load_x(a)
        kcu, yc = load_cal_y(a, dp, dg, ngs)
        assert np.array_equal(kcu, keys_u), f"{a}: cal/raw 键集不一致"
        del kcu, x
        cg_loc = (keys_u % ngs).astype(np.int64)
        cp_loc = (keys_u // ngs).astype(np.int64)
        gstr = dg.to_pandas().to_numpy()[cg_loc]
        cg_glob = vocab_g.get_indexer(pd.Index(gstr.astype(str)))
        assert (cg_glob >= 0).all(), f"{a}: 全局基因词汇表缺键"
        nT = int(cp_loc.max()) + 1
        M = np.zeros((nT, NG), dtype=np.float64)
        M[cp_loc, cg_glob] = yc.astype(np.float64)
        del yc, keys_u, cg_loc
        # ---- v3 basis: u1=行均值方向 + 去均值 top-K 右奇异向量 ----
        # (RR^T eigh 与 SVD Vt 数学等价, 大 nT 下更快)
        mm = M.mean(0)
        u1 = mm / max(np.linalg.norm(mm), 1e-12)
        R = M - mm[None, :]
        G = R @ R.T
        w, Vm = np.linalg.eigh(G)
        order = np.argsort(-w)[:K]
        basis = [u1]
        for i in order:
            s_i = float(np.sqrt(max(w[i], 0.0)))
            if s_i <= 1e-12:
                continue
            basis.append((R.T @ Vm[:, i] / s_i).astype(np.float64))
        U = np.vstack(basis)
        # transform (与 train_v3_probe.transform 同式): P' = M − β Σ_k (M·u_k) u_k
        Pt = M - BETA * (M @ U.T) @ U
        vals = Pt[cp_loc, cg_glob]
        del M, Pt, G, R, mm
        assert np.isfinite(vals).all()
        out = pd.DataFrame({
            "target_gene": dp.to_pandas().to_numpy()[cp_loc],
            "gene": gstr,
            "logfc": vals.astype(np.float32),
        })
        out.to_parquet(f"{SIG}/{a}_calV3.parquet", index=False)
        n_out = len(out)
        del out, vals, dp, dg
        print(f"[{a}] targets={nT:,} rows={n_out:,} width={NG} K={K} beta={BETA} "
              f"t={time.time()-t0:.0f}s", flush=True)
    print(f"done t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
