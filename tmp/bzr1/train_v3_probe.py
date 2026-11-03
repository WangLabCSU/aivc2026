#!/usr/bin/env python
"""bzr1 v3 probe — 从已训练 M_ctx_v3 直接读数, 零训练成本.

回答 v3 FAIL 的两个疑点:
(1) sim5 的 panel-reversal (校准信号集中在高信号坐标) 在 v3 的 hold 数据下是否复现;
(2) (K, β) 全网格 rank-pds × spearman 权衡面: 有无两全点 (pds>NC 且 sp≈rawcal).
特征构造/统计与 train_calibrator_v3.py 完全一致 (TGSD/TGQ9 真实特征).
"""
import json, time
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr, rankdata

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
CTX = ["gwps", "hct116_all", "hepg2_all", "jurkat_all", "rpe1_all", "k562_ess_all"]


def main():
    t0 = time.time()
    genes, perts = set(), set()
    for n in CTX:
        df = pd.read_parquet(f"{SIG}/{n}.parquet", columns=["target_gene", "gene"])
        genes |= set(pd.unique(df["gene"].astype(str)))
        perts |= set(pd.unique(df["target_gene"].astype(str)))
        del df
    vocab_g = pd.Index(sorted(genes)); vocab_p = pd.Index(sorted(perts))
    NG = len(vocab_g); del genes, perts
    print(f"vocab g={NG} p={len(vocab_p)} t={time.time()-t0:.0f}s", flush=True)

    bench = pd.read_csv(f"{ROOT}/outputs/metrics/local_bench/dualdeep_lam70mf14_Q/Q_HepG2_eval/results.csv")
    hold_codes = np.where(vocab_p.isin(set(bench["perturbation"].astype(str))))[0]
    assert len(hold_codes) == 100

    def load_keys(name):
        df = pd.read_parquet(f"{SIG}/{name}.parquet", columns=["target_gene", "gene", "logfc"])
        cp = vocab_p.get_indexer(df["target_gene"].astype(str).values)
        cg = vocab_g.get_indexer(df["gene"].astype(str).values)
        keys = cp.astype(np.int64) * NG + cg.astype(np.int64)
        lf = df["logfc"].values.astype(np.float32); del df
        keys_u, inv = np.unique(keys, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(keys_u)).astype(np.float64)
        ssum = np.bincount(inv, weights=lf.astype(np.float64), minlength=len(keys_u))
        lf = (ssum / cnt).astype(np.float32); del keys, inv
        return keys_u, lf

    data = {}
    for n in CTX:
        data[n] = load_keys(n)
    print(f"data loaded t={time.time()-t0:.0f}s", flush=True)

    # tg stats: hepg2_all, hold perts 排除 (同 v3)
    keys, lf = data["hepg2_all"]
    cp0 = keys // NG; cg0 = keys % NG
    m = ~np.isin(cp0, hold_codes)
    dfg = pd.DataFrame({"cg": cg0[m], "lf": lf[m], "alf": np.abs(lf[m])})
    TGM = np.full(NG, 0.0, dtype=np.float32)
    TGS = np.full(NG, 0.09, dtype=np.float32)
    TGQ = np.full(NG, 0.05, dtype=np.float32)
    tgm = dfg.groupby("cg")["lf"].mean(); tgs = dfg.groupby("cg")["lf"].std().fillna(0.09)
    tgq = dfg.groupby("cg")["alf"].quantile(0.9)
    TGM[np.asarray(tgm.index)] = tgm.values.astype(np.float32)
    TGS[np.asarray(tgs.index)] = tgs.values.astype(np.float32)
    TGQ[np.asarray(tgq.index)] = tgq.values.astype(np.float32)
    del dfg, tgm, tgs, tgq, keys, lf, cp0, cg0, m
    print(f"tg stats t={time.time()-t0:.0f}s", flush=True)

    SRC = [c for c in CTX if c != "hepg2_all"]
    SRCID = {c: i for i, c in enumerate(SRC)}
    GME = {}; GST = {}; GQ9 = {}; PME = {}; PQ9 = {}
    for n in CTX:
        kk, ll = data[n]
        cpp = kk // NG; cgg = kk % NG
        df = pd.DataFrame({"cp": cpp, "cg": cgg, "lf": ll, "alf": np.abs(ll)})
        gme = df.groupby("cg")["lf"].mean(); gst = df.groupby("cg")["lf"].std().fillna(1.0)
        gq9 = df.groupby("cg")["alf"].quantile(0.9)
        pme = df.groupby("cp")["alf"].mean(); pq9 = df.groupby("cp")["alf"].quantile(0.9)

        def _fill(ser, length, default):
            arr = np.full(length, default, dtype=np.float32)
            arr[np.asarray(ser.index, dtype=np.int64)] = ser.values.astype(np.float32)
            return arr
        GME[n] = _fill(gme, NG, 0.0); GST[n] = _fill(gst, NG, 1.0); GQ9[n] = _fill(gq9, NG, 0.05)
        PME[n] = _fill(pme, len(vocab_p), 0.05); PQ9[n] = _fill(pq9, len(vocab_p), 0.05)
        del df, gme, gst, gq9, pme, pq9
    print(f"stats done t={time.time()-t0:.0f}s", flush=True)

    def pair_hold(a):
        ka, xa = data[a]; kb, xb = data["hepg2_all"]
        shared, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
        x = xa[ia]; y = xb[ib]
        cgsh = shared % NG; cpsh = shared // NG
        hold = np.isin(cpsh, hold_codes)
        X = np.column_stack([
            x[hold], np.abs(x[hold]), GME[a][cgsh[hold]], GST[a][cgsh[hold]], GQ9[a][cgsh[hold]],
            PME[a][cpsh[hold]], PQ9[a][cpsh[hold]],
            (x[hold] / GST[a][cgsh[hold]]).astype(np.float32),
            (x[hold] / np.maximum(PQ9[a][cpsh[hold]], 1e-3)).astype(np.float32),
            (x[hold] / np.maximum(GQ9[a][cgsh[hold]], 1e-3)).astype(np.float32),
            TGM[cgsh[hold]], TGS[cgsh[hold]], TGQ[cgsh[hold]],
            np.full(int(hold.sum()), SRCID[a], dtype=np.float32),
        ]).astype(np.float32)
        return X, x[hold], y[hold], cpsh[hold], cgsh[hold], shared[hold]

    FEATS = ["x", "ax", "gmean", "gstd", "gq90", "pmean", "pq90", "x_z", "x_rel_p", "x_rel_g",
             "tg_mean", "tg_std", "tg_q90", "src_id"]
    Xh_l = []; xh_l = []; yh_l = []; kh_l = []
    for a in SRC:
        X, x, y, cpsh, cgsh, sh = pair_hold(a)
        Xh_l.append(X); xh_l.append(x); yh_l.append(y); kh_l.append(sh)
        print(f"  hold {a}: {X.shape[0]:,} t={time.time()-t0:.0f}s", flush=True)
        del X, x, y, cpsh, cgsh, sh
    X_hold = np.concatenate(Xh_l); x_hold = np.concatenate(xh_l)
    y_hold = np.concatenate(yh_l); keys_hold = np.concatenate(kh_l)
    del Xh_l, xh_l, yh_l, kh_l
    print(f"hold {X_hold.shape} t={time.time()-t0:.0f}s", flush=True)

    bst = xgb.Booster(); bst.load_model(f"{ROOT}/tmp/bzr1/M_ctx_v3.json")
    pred_hold = bst.predict(xgb.DMatrix(X_hold, feature_names=FEATS))
    del X_hold
    print(f"pred done t={time.time()-t0:.0f}s", flush=True)

    # 行矩阵: 键唯一化 (跨源均值), 全基因宽度
    def build(vals):
        uk, inv = np.unique(keys_hold, return_inverse=True)
        cnt = np.bincount(inv).astype(np.float64)
        ssum = np.bincount(inv, weights=vals.astype(np.float64))
        avg = ssum / cnt
        rows = uk // NG; cols = uk % NG
        ur, inv_r = np.unique(rows, return_inverse=True)
        M = np.zeros((len(ur), NG), dtype=np.float64)
        M[inv_r, cols] = avg
        return M
    P0 = build(pred_hold.astype(np.float64))
    X0 = build(x_hold.astype(np.float64))
    Y0 = build(y_hold.astype(np.float64))
    print(f"matrices {P0.shape} t={time.time()-t0:.0f}s", flush=True)

    def rank_pds(M, Y):
        Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
        Yn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        D = 1.0 - Mn @ Yn.T
        rk = np.apply_along_axis(rankdata, 1, D)
        return float(np.mean(1.0 - np.diag(rk) / D.shape[1]))

    def sp_med(M, Y):
        sps = [spearmanr(M[i], Y[i]).statistic for i in range(M.shape[0])
               if np.std(M[i]) > 0 and np.std(Y[i]) > 0]
        return float(np.median(sps))

    top500 = np.argsort(-np.abs(Y0).mean(0))[:500]
    print(f"panel500: NC={rank_pds(X0[:, top500], Y0[:, top500]):.4f} "
          f"rawcal={rank_pds(P0[:, top500], Y0[:, top500]):.4f}", flush=True)

    def fit_basis(Ma, K):
        mm = Ma.mean(0); mm /= max(np.linalg.norm(mm), 1e-12)
        if K <= 0:
            return [mm]
        R = Ma - Ma.mean(0, keepdims=True)
        _, S, Vt = np.linalg.svd(R, full_matrices=False)
        return [mm] + [Vt[i] for i in range(min(K, len(S)))]

    def transform(M, basis, beta):
        out = M.copy()
        for u in basis:
            out = out - beta * np.outer(M @ u, u)
        return out

    results = {}
    for K in (0, 4, 8, 16, 32):
        basis = fit_basis(P0, K)
        for b in (0.25, 0.5, 0.75, 1.0):
            Pt = transform(P0, basis, b)
            results[f"K{K}_b{b}"] = {"pds": rank_pds(Pt, Y0), "sp": sp_med(Pt, Y0)}
            r = results[f"K{K}_b{b}"]
            print(f"  K={K:>2} b={b:.2f}: pds={r['pds']:.4f} sp={r['sp']:.4f}", flush=True)
    pds_nc, sp_nc = rank_pds(X0, Y0), sp_med(X0, Y0)
    pds_raw, sp_raw = rank_pds(P0, Y0), sp_med(P0, Y0)
    print(f"NC    : pds={pds_nc:.4f} sp={sp_nc:.4f}", flush=True)
    print(f"rawcal: pds={pds_raw:.4f} sp={sp_raw:.4f}", flush=True)
    both = [k for k, v in results.items()
            if v["pds"] > max(pds_nc, pds_raw) and v["sp"] >= sp_nc - 0.02]
    print(f"TWO-WIN points (pds>max(nc,raw) and sp>=nc-0.02): {both}", flush=True)
    with open(f"{ROOT}/tmp/bzr1/probe_v4_meta.json", "w") as f:
        json.dump({"pds_nc": pds_nc, "sp_nc": sp_nc, "pds_raw": pds_raw, "sp_raw": sp_raw,
                   "grid": results, "two_win": both}, f, indent=2, default=float)
    print(f"done t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
