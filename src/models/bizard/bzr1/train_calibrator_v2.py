#!/usr/bin/env python
"""bzr1 v2 — 直接预测 y (弃残差公式) + 目标侧基因先验特征 (iter030-修正2).

v1 判决 (train_meta.json): pred = x + r̂ 三门全败; 残差学习器真 (corr 0.586) 但 x 方差主导.
v2 假设: 模型直接拟合 y, 特征加入 (i) 目标侧基因级统计 (hepg2 train perts only,
  bench-100 留出 — 基因响应先验可跨 pert 泛化) (ii) 源上下文 id.
诚实基线 (hold 上直接读数): direct transfer / zero / gene-prior (TG_ME).
过门判据 (预注册, 同 v1): corr(pred,y) > corr(x,y) 且 MAE(pred,y) < mean|y| 且 spearman med > direct med.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

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

    # ---- 词汇表 ----
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
        # iter030-修正3 对齐修复: np.unique 排序键后值必须经 inv 跟随重排, 无条件执行
        # (旧代码仅在有重复行时去重; 无重复时跳过 → 键排序而值留文件序, corr 归零)
        cnt = np.bincount(inv, minlength=len(keys_u)).astype(np.float64)
        ssum = np.bincount(inv, weights=lf.astype(np.float64), minlength=len(keys_u))
        lf = (ssum / cnt).astype(np.float32)
        del keys, inv
        return keys_u, lf

    data = {}
    for n in CTX:
        data[n] = load_keys(n)
        print(f"loaded {n}: {len(data[n][0]):,} t={time.time()-t0:.0f}s", flush=True)

    # ---- 源侧统计 (hepg2 剔 hold) + 目标侧基因先验 (hepg2 train perts only) ----
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
        if n == "hepg2_all":  # 目标侧基因先验: train perts only
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

    # ---- 配对特征: v1 10 列 + 目标侧基因先验 3 列 + src_id ----
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
    # 基因先验基线: 目标侧 train-pert 基因均值, 零模型
    gp = TGM[cg_hold]
    mae_g, corr_g, sp_g, _ = metrics(gp, y_hold, cp_hold)
    print(f"BASELINE direct:    mae={mae0:.4f} corr={corr0:.4f} spearman_med={sp0:.4f}", flush=True)
    print(f"BASELINE zero:      mae={mae_z:.4f}", flush=True)
    print(f"BASELINE gene-prior: mae={mae_g:.4f} corr={corr_g:.4f} spearman_med={sp_g:.4f}", flush=True)

    dtr = xgb.DMatrix(X_ctx, label=y_ctx, feature_names=FEATS)
    bst = xgb.train(params, dtr, num_boost_round=args.rounds,
                    evals=[(dtr, "train")], verbose_eval=100)
    del dtr, X_ctx, y_ctx
    bst.save_model(f"{args.out}/M_ctx_v2.json")

    dh = xgb.DMatrix(X_hold, feature_names=FEATS)
    pred = bst.predict(dh)
    del dh, X_hold

    mae1, corr1, sp1, n1 = metrics(pred, y_hold, cp_hold)
    gates = {"corr_beats_direct": corr1 > corr0,
             "mae_beats_zero": mae1 < mae_z,
             "spearman_beats_direct": sp1 > sp0}
    verdict = "PASS" if all(gates.values()) else "FAIL"
    meta = {"t_sec": round(time.time() - t0, 1), "rounds": args.rounds, "seed": args.seed,
            "params": params, "feats": FEATS, "formulation": "direct-y + target-gene-prior feats",
            "n_train": int(len(pred)), "n_hold": int(len(y_hold)),
            "baseline_direct": {"mae": mae0, "corr": corr0, "spearman_med": sp0},
            "baseline_zero_mae": mae_z,
            "baseline_gene_prior": {"mae": mae_g, "corr": corr_g, "spearman_med": sp_g},
            "model_pred": {"mae": mae1, "corr": corr1, "spearman_med": sp1},
            "gates": gates, "verdict": verdict}
    with open(f"{args.out}/train_meta_v2.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"MODEL direct-y pred: mae={mae1:.4f} corr={corr1:.4f} spearman_med={sp1:.4f}", flush=True)
    print(f"GATES: {gates}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
