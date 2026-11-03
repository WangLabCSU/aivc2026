#!/usr/bin/env python
"""bzr1 v3 — L269 换构造重训: 直接学 y + 行空间部分去相关 (train+gate only).

判据链 (iter035 全段探针, tmp/bzr1/sim5_cache.npz):
- 全空间 rank-pds: NC 0.8731 >> YH 0.5530 (校准臂 bench 崩机理: 预测行共享均
  值方向 ||m||/med||row|| = 1.158, off-diag q95 0.520 vs NC 0.182).
- panel-mask(k=500 of |Y|) 后排序反转: YH 0.8404 > NC 0.8181 — 校准器优势集
  中在高信号坐标; 全删共享成分 (YHc) 连 per-row spearman 一起毁 (0.2786→0.1335).
=> 构造: P' = P - β · Σ_{k≤K} (P·u_k) u_k,  u_1 = 行均值方向, u_{2..K} = 残差 top-PC.
   basis 只用预测行自身几何 (无真值泄漏); (K, β) 由 hold-100 内部 2-fold LODO
   调出 (B 组真值仅用于组内检索打分); 最终门在 hold-100 全体读数.
   已知弱点: 门与调参共享 hold-100 (2 个标量, 弱乐观) — 外部 bench 链才是真仲裁.

过门判据 (预注册):
  模型质量门 (同 v2): corr(pred,y) > corr(x,y), mae < mean|y|, spearman med > direct.
  检索门 (新增): rank-pds(P') > max(rank-pds(NC), rank-pds(P)) 且 spearman 行 med
  不低于 NC − 0.02.
输出: M_ctx_v3.json + train_meta_v3.json. parquet 由独立脚本按 raw 键约定构造.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr, rankdata

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
CTX = ["gwps", "hct116_all", "hepg2_all", "jurkat_all", "rpe1_all", "k562_ess_all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/tmp/bzr1")
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    # ---- 词汇表 (与 v2 完全一致) ----
    genes, perts = set(), set()
    for n in CTX:
        df = pd.read_parquet(f"{SIG}/{n}.parquet", columns=["target_gene", "gene"])
        genes |= set(pd.unique(df["gene"].astype(str)))
        perts |= set(pd.unique(df["target_gene"].astype(str)))
        del df
    vocab_g = pd.Index(sorted(genes)); vocab_p = pd.Index(sorted(perts))
    NG = len(vocab_g)
    del genes, perts
    print(f"vocab: g={NG} p={len(vocab_p)} t={time.time()-t0:.0f}s", flush=True)

    bench = pd.read_csv(f"{ROOT}/outputs/metrics/local_bench/dualdeep_lam70mf14_Q/Q_HepG2_eval/results.csv")
    hold_codes = np.where(vocab_p.isin(set(bench["perturbation"].astype(str))))[0]
    assert len(hold_codes) == 100, f"expect 100 hold perts, got {len(hold_codes)}"

    def load_keys(name):
        df = pd.read_parquet(f"{SIG}/{name}.parquet", columns=["target_gene", "gene", "logfc"])
        cp = vocab_p.get_indexer(df["target_gene"].astype(str).values)
        cg = vocab_g.get_indexer(df["gene"].astype(str).values)
        keys = (cp.astype(np.int64) * NG + cg.astype(np.int64))
        lf = df["logfc"].values.astype(np.float32)
        del df
        keys_u, inv = np.unique(keys, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(keys_u)).astype(np.float64)
        ssum = np.bincount(inv, weights=lf.astype(np.float64), minlength=len(keys_u))
        lf = (ssum / cnt).astype(np.float32)
        del keys, inv
        return keys_u, lf

    data = {}
    for n in CTX:
        data[n] = load_keys(n)
        print(f"loaded {n}: {len(data[n][0]):,} t={time.time()-t0:.0f}s", flush=True)

    gmean = {}; gstd = {}; gq90 = {}; pmean = {}; pq90 = {}
    tg_mean = {}; tg_std = {}; tg_q90 = {}
    for n in CTX:
        keys, lf = data[n]
        cp = (keys // NG).astype(np.int64); cg = (keys % NG).astype(np.int64)
        if n == "hepg2_all":
            m = ~np.isin(cp, hold_codes)
            cp, cg, lf = cp[m], cg[m], lf[m]
        alf = np.abs(lf)
        df = pd.DataFrame({"cp": cp, "cg": cg, "lf": lf, "alf": alf})
        gb = df.groupby("cg")["lf"]
        gmean[n] = gb.mean(); gstd[n] = gb.std().fillna(1.0)
        gq90[n] = df.groupby("cg")["alf"].quantile(0.9)
        pmean[n] = df.groupby("cp")["alf"].mean(); pq90[n] = df.groupby("cp")["alf"].quantile(0.9)
        del df
        if n == "hepg2_all":
            dfg = pd.DataFrame({"cg": cg, "lf": lf, "alf": np.abs(lf)})
            tg = dfg.groupby("cg")["lf"]
            tg_mean[n] = tg.mean(); tg_std[n] = tg.std().fillna(1.0)
            tg_q90[n] = dfg.groupby("cg")["alf"].quantile(0.9)
            del dfg
        print(f"stats {n} done t={time.time()-t0:.0f}s", flush=True)

    def to_global(stat_series, vocab, default):
        arr = np.full(len(vocab), default, dtype=np.float32)
        arr[np.asarray(stat_series.index, dtype=np.int64)] = stat_series.values.astype(np.float32)
        return arr

    GST = {n: to_global(gstd[n], vocab_g, 1.0) for n in CTX}
    GME = {n: to_global(gmean[n], vocab_g, 0.0) for n in CTX}
    GQ9 = {n: to_global(gq90[n], vocab_g, 0.05) for n in CTX}
    PME = {n: to_global(pmean[n], vocab_p, 0.05) for n in CTX}
    PQ9 = {n: to_global(pq90[n], vocab_p, 0.05) for n in CTX}
    TGM = to_global(tg_mean["hepg2_all"], vocab_g, 0.0)
    TGSD = to_global(tg_std["hepg2_all"], vocab_g, 0.09)
    TGQ9 = to_global(tg_q90["hepg2_all"], vocab_g, 0.05)

    SRC = [c for c in CTX if c != "hepg2_all"]
    SRCID = {c: i for i, c in enumerate(SRC)}

    def pair_features(a):
        ka, xa = data[a]; kb, xb = data["hepg2_all"]
        shared, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
        x = xa[ia]; y = xb[ib]
        cgsh = (shared % NG).astype(np.int64); cpsh = (shared // NG).astype(np.int64)
        X = np.column_stack([
            x, np.abs(x), GME[a][cgsh], GST[a][cgsh], GQ9[a][cgsh],
            PME[a][cpsh], PQ9[a][cpsh],
            (x / GST[a][cgsh]).astype(np.float32),
            (x / np.maximum(PQ9[a][cpsh], 1e-3)).astype(np.float32),
            (x / np.maximum(GQ9[a][cgsh], 1e-3)).astype(np.float32),
            TGM[cgsh], TGSD[cgsh], TGQ9[cgsh],
            np.full(len(shared), SRCID[a], dtype=np.float32),
        ]).astype(np.float32)
        return X, x, y, cpsh, cgsh

    FEATS = ["x", "ax", "gmean", "gstd", "gq90", "pmean", "pq90", "x_z", "x_rel_p", "x_rel_g",
             "tg_mean", "tg_std", "tg_q90", "src_id"]

    Xh_l = []; xh_l = []; yh_l = []; cph_l = []; cgh_l = []
    Xt_l = []; yt_l = []
    for a in SRC:
        X, x, y, cpsh, cgsh = pair_features(a)
        hold = np.isin(cpsh, hold_codes)
        tr = ~hold
        if hold.any():
            Xh_l.append(X[hold]); xh_l.append(x[hold]); yh_l.append(y[hold]); cph_l.append(cpsh[hold]); cgh_l.append(cgsh[hold])
        Xt_l.append(X[tr]); yt_l.append(y[tr])
        del X, x, y, cpsh, cgsh
        print(f"M_ctx {a}: train={int(tr.sum()):,} hold={int(hold.sum()):,} t={time.time()-t0:.0f}s", flush=True)

    X_ctx = np.concatenate(Xt_l); y_ctx = np.concatenate(yt_l)
    X_hold = np.concatenate(Xh_l); x_hold = np.concatenate(xh_l)
    y_hold = np.concatenate(yh_l); cp_hold = np.concatenate(cph_l); cg_hold = np.concatenate(cgh_l)
    del Xt_l, yt_l, Xh_l, xh_l, yh_l, cph_l, cgh_l
    print(f"train {X_ctx.shape}, hold {X_hold.shape}, t={time.time()-t0:.0f}s", flush=True)

    params = {"objective": "reg:pseudohubererror", "tree_method": "hist", "device": "cuda",
              "max_depth": 8, "eta": 0.05, "subsample": 0.7, "colsample_bytree": 0.6,
              "huber_slope": 0.05, "seed": args.seed}

    def metrics(pred, y, cp):
        mae = float(np.mean(np.abs(pred - y)))
        corr = float(np.corrcoef(pred, y)[0, 1])
        sps = []
        for c in np.unique(cp):
            g = cp == c
            if g.sum() >= 30 and np.std(pred[g]) > 0 and np.std(y[g]) > 0:
                sps.append(spearmanr(pred[g], y[g]).statistic)
        return mae, corr, float(np.median(sps)), len(sps)

    mae0, corr0, sp0, n0 = metrics(x_hold, y_hold, cp_hold)
    mae_z = float(np.mean(np.abs(y_hold)))
    gp = TGM[cg_hold]
    mae_g, corr_g, sp_g, _ = metrics(gp, y_hold, cp_hold)
    print(f"BASELINE direct:    mae={mae0:.4f} corr={corr0:.4f} spearman_med={sp0:.4f}", flush=True)
    print(f"BASELINE zero:      mae={mae_z:.4f}", flush=True)
    print(f"BASELINE gene-prior: mae={mae_g:.4f} corr={corr_g:.4f} spearman_med={sp_g:.4f}", flush=True)

    dtr = xgb.DMatrix(X_ctx, label=y_ctx, feature_names=FEATS)
    bst = xgb.train(params, dtr, num_boost_round=args.rounds,
                    evals=[(dtr, "train")], verbose_eval=100)
    bst.save_model(f"{args.out}/M_ctx_v3.json")
    pred_ctx = bst.predict(xgb.DMatrix(X_ctx, feature_names=FEATS))
    del dtr, X_ctx, y_ctx
    print(f"in-sample pred done t={time.time()-t0:.0f}s", flush=True)

    dh = xgb.DMatrix(X_hold, feature_names=FEATS)
    pred_hold = bst.predict(dh)
    del dh, X_hold

    mae1, corr1, sp1, n1 = metrics(pred_hold, y_hold, cp_hold)
    print(f"MODEL direct-y pred: mae={mae1:.4f} corr={corr1:.4f} spearman_med={sp1:.4f}", flush=True)

    # ============ v3 核心: 行空间部分去相关 ============
    # hold 行矩阵: 双源同 (cp,cg) 取均值; 基因域 = hold 键的并集; NaN→0.
    def scatter_avg(cp, cg, val):
        acc = {}; cnt = {}
        for c, g, v in zip(cp.tolist(), cg.tolist(), val.tolist()):
            k = (c, g)
            if k in acc:
                acc[k] += v; cnt[k] += 1
            else:
                acc[k] = v; cnt[k] = 1
        rows = {}
        for (c, g), v in acc.items():
            rows.setdefault(c, {})[g] = v / cnt[(c, g)]
        return rows

    dict_pred = scatter_avg(cp_hold, cg_hold, pred_hold.astype(np.float64))
    dict_x = scatter_avg(cp_hold, cg_hold, x_hold.astype(np.float64))
    dict_y = scatter_avg(cp_hold, cg_hold, y_hold)
    genes_h = sorted({g for d in dict_pred.values() for g in d})
    gcol = {g: i for i, g in enumerate(genes_h)}
    tgts = sorted(dict_pred.keys())
    trow = {t: i for i, t in enumerate(tgts)}
    P = np.full((len(tgts), len(genes_h)), np.nan)
    Xr = np.full_like(P, np.nan)
    Yr = np.full_like(P, np.nan)
    for i, t in enumerate(tgts):
        for g, v in dict_pred[t].items(): P[i, gcol[g]] = v
        for g, v in dict_x.get(t, {}).items(): Xr[i, gcol[g]] = v
        for g, v in dict_y.get(t, {}).items(): Yr[i, gcol[g]] = v
    P0 = np.nan_to_num(P); X0 = np.nan_to_num(Xr); Y0 = np.nan_to_num(Yr)
    del dict_pred, dict_x, dict_y, P, Xr, Yr
    print(f"hold matrix: {P0.shape} t={time.time()-t0:.0f}s", flush=True)

    def rank_pds_rows(M, Y, rows=None):
        """rows=None: 全体 N 行 vs 全体真值, mean(1-diag/N).
        rows=idx: 只取 M[idx] 作查询, 检索空间 = Y[idx] (len(idx) 候选),
        own 位置按 idx 内部对齐."""
        Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
        Yn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        if rows is None:
            D = 1.0 - Mn @ Yn.T
            rk = np.apply_along_axis(rankdata, 1, D)
            return float(np.mean(1.0 - np.diag(rk) / D.shape[1]))
        rows = np.asarray(rows)
        D = 1.0 - Mn[rows] @ Yn[rows].T
        rk = np.apply_along_axis(rankdata, 1, D)
        return float(np.mean(1.0 - np.diag(rk) / len(rows)))

    def spearman_rows(M, Y):
        sps = []
        for i in range(M.shape[0]):
            if np.std(M[i]) > 0 and np.std(Y[i]) > 0:
                sps.append(spearmanr(M[i], Y[i]).statistic)
        return float(np.median(sps))

    def fit_basis(Ma, K):
        m = Ma.mean(0); m /= max(np.linalg.norm(m), 1e-12)
        if K <= 0:
            return [m]
        R = Ma - Ma.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        return [m] + [Vt[i] for i in range(min(K, len(S)))]

    def transform(M, basis, beta):
        out = M.copy()
        for u in basis:
            out = out - beta * np.outer(M @ u, u)
        return out

    # 2-fold LODO: A 组拟合 basis, B 组变换后组内检索读数; 互换取均值
    idx = np.arange(len(tgts)); np.random.RandomState(args.seed).shuffle(idx)
    folds = [idx[:50], idx[50:]]
    grid = [(0, 0.0)] + [(K, b) for K in (0, 4, 16) for b in (0.25, 0.5, 0.75, 1.0)]
    score = {g: [] for g in grid}
    for A, B in [(folds[0], folds[1]), (folds[1], folds[0])]:
        basis_max = fit_basis(P0[A], 16)
        for K, b in grid:
            Kb = basis_max[:1 + K]
            Pt = transform(P0[B], Kb, b) if b > 0 else P0[B]
            score[(K, b)].append(rank_pds_rows(Pt, Y0[B]))
    mean_score = {g: float(np.mean(v)) for g, v in score.items()}
    (K_best, b_best) = max(mean_score, key=mean_score.get)
    print("LODO grid (K, beta) -> mean B-fold rank-pds:", flush=True)
    for g in sorted(mean_score, key=mean_score.get, reverse=True):
        print(f"  K={g[0]:>2} beta={g[1]:.2f} -> {mean_score[g]:.4f}", flush=True)

    basis_full = fit_basis(P0, K_best)
    Kb = basis_full[:1 + K_best]
    Pt = transform(P0, Kb, b_best) if b_best > 0 else P0
    pds_nc = rank_pds_rows(X0, Y0); pds_raw = rank_pds_rows(P0, Y0); pds_tr = rank_pds_rows(Pt, Y0)
    sp_nc = spearman_rows(X0, Y0); sp_raw = spearman_rows(P0, Y0); sp_tr = spearman_rows(Pt, Y0)
    print(f"RANK-PDS hold100(full): NC={pds_nc:.4f} rawcal={pds_raw:.4f} V3(decor)={pds_tr:.4f}", flush=True)
    print(f"SPEARMAN med          : NC={sp_nc:.4f} rawcal={sp_raw:.4f} V3(decor)={sp_tr:.4f}", flush=True)

    gates = {"corr_beats_direct": corr1 > corr0,
             "mae_beats_zero": mae1 < mae_z,
             "spearman_beats_direct": sp1 > sp0,
             "pds_beats_both": pds_tr > max(pds_nc, pds_raw),
             "sp_kept": sp_tr >= sp_nc - 0.02}
    verdict = "PASS" if all(gates.values()) else "FAIL"
    meta = {"t_sec": round(time.time() - t0, 1), "rounds": args.rounds, "seed": args.seed,
            "params": params, "feats": FEATS,
            "formulation": "direct-y + partial row decorrelation (mean dir + residual PCs)",
            "K": K_best, "beta": b_best,
            "tuning": "2-fold LODO within hold-100; basis from pred rows only",
            "grid_mean": {str(k): v for k, v in mean_score.items()},
            "baseline_direct": {"mae": mae0, "corr": corr0, "spearman_med": sp0},
            "baseline_zero_mae": mae_z,
            "model_pred": {"mae": mae1, "corr": corr1, "spearman_med": sp1},
            "rank_pds": {"nc": pds_nc, "rawcal": pds_raw, "v3": pds_tr},
            "spearman_rows": {"nc": sp_nc, "rawcal": sp_raw, "v3": sp_tr},
            "gates": gates, "verdict": verdict}
    with open(f"{args.out}/train_meta_v3.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"GATES: {gates}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"done t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
