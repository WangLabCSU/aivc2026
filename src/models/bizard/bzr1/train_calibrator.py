#!/usr/bin/env python
"""bzr1 — bizard 重设计第一臂: 学习式跨上下文签名校准器 (iter030, 用户指令 2026-10-17).

任务: 源上下文签名 logfc(x) → 目标上下文签名 logfc(y), 同一 (pert,gene) matched pairs.
设计: 特征只用源签名自身 + 基因/扰动级源侧协变量 (推理时可算, 无目标泄漏).
目标: residual r = y - x; 校正签名 = x + r̂.
诚实验证: Q_HepG2 bench-100 靶永远留出 (hepg2_all.parquet 即 Q truth 文件, iter028 陷阱);
  hepg2_all 侧基因/扰动统计同样剔除 bench-100 靶后计算.
基线判据 (探针 bn0l9qc37): direct transfer MAE 0.0658 / zero-baseline MAE 0.0577 /
  per-pert spearman med 0.0976 / corr 0.2142.
过门判据 (预注册): corr(pred,y) > corr(x,y) 且 MAE(pred,y) < mean|y| 且 spearman med > direct med.
"""
import argparse, json, time
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
    import os
    os.makedirs(args.out, exist_ok=True)

    # ---- 词汇表 (两遍读: 先收集 unique, 再编码) ----
    genes, perts = set(), set()
    for n in CTX:
        df = pd.read_parquet(f"{SIG}/{n}.parquet", columns=["target_gene", "gene"])
        genes |= set(pd.unique(df["gene"].astype(str)))
        perts |= set(pd.unique(df["target_gene"].astype(str)))
        del df
        print(f"vocab after {n}: genes={len(genes)} perts={len(perts)} t={time.time()-t0:.0f}s", flush=True)
    vocab_g = pd.Index(sorted(genes)); vocab_p = pd.Index(sorted(perts))
    NG = len(vocab_g)
    del genes, perts

    # bench-100 hold perts (hepg2 侧全部剔除, iter028 陷阱)
    bench = pd.read_csv(f"{ROOT}/outputs/metrics/local_bench/dualdeep_lam70mf14_Q/Q_HepG2_eval/results.csv")
    hold_perts = set(bench["perturbation"].astype(str))
    hold_codes = np.where(vocab_p.isin(hold_perts))[0]
    assert len(hold_codes) == 100, f"expect 100 hold perts, got {len(hold_codes)}"
    print("hold perts:", len(hold_codes), flush=True)

    # ---- 载入各 ctx: 去重后的 unique sorted keys ----
    def load_keys(name):
        df = pd.read_parquet(f"{SIG}/{name}.parquet", columns=["target_gene", "gene", "logfc"])
        cp = vocab_p.get_indexer(df["target_gene"].astype(str).values)
        cg = vocab_g.get_indexer(df["gene"].astype(str).values)
        assert (cp >= 0).all() and (cg >= 0).all(), f"{name}: unseen keys?!"
        keys = (cp.astype(np.int64) * NG + cg.astype(np.int64))
        lf = df["logfc"].values.astype(np.float32)
        del df
        keys_u, inv = np.unique(keys, return_inverse=True)
        if len(keys_u) != len(keys):
            cnt = np.bincount(inv, minlength=len(keys_u)).astype(np.float64)
            ssum = np.bincount(inv, weights=lf.astype(np.float64), minlength=len(keys_u))
            lf = (ssum / cnt).astype(np.float32)
            print(f"{name}: dedup {len(keys):,} -> {len(keys_u):,} (mult {len(keys)/len(keys_u):.4f})", flush=True)
        del keys, inv
        return keys_u, lf  # unique, sorted

    data = {}
    for n in CTX:
        data[n] = load_keys(n)
        print(f"loaded {n}: {len(data[n][0]):,} unique rows t={time.time()-t0:.0f}s", flush=True)

    # ---- 源侧统计 (hepg2_all 剔除 hold 靶行后计算) ----
    gmean = {}; gstd = {}; gq90 = {}; pmean = {}; pq90 = {}
    for n in CTX:
        keys, lf = data[n]
        cp = (keys // NG).astype(np.int64); cg = (keys % NG).astype(np.int64)
        if n == "hepg2_all":
            m = ~np.isin(cp, hold_codes)
            cp, cg, lf = cp[m], cg[m], lf[m]
        alf = np.abs(lf)
        df = pd.DataFrame({"cp": cp, "cg": cg, "lf": lf, "alf": alf})
        del alf
        gb = df.groupby("cg")["lf"]
        gmean[n] = gb.mean()
        gstd[n] = gb.std().fillna(1.0)
        gq90[n] = df.groupby("cg")["alf"].quantile(0.9)
        pmean[n] = df.groupby("cp")["alf"].mean()
        pq90[n] = df.groupby("cp")["alf"].quantile(0.9)
        del df
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

    # ---- 组装配对特征 (int64 key join, 无 pandas merge) ----
    def pair_features(a, b):
        ka, xa = data[a]
        kb, xb = data[b]
        shared, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
        x = xa[ia]; y = xb[ib]
        cgsh = (shared % NG).astype(np.int64); cpsh = (shared // NG).astype(np.int64)
        gs_ = GST[a][cgsh]; gm_ = GME[a][cgsh]; gq_ = GQ9[a][cgsh]
        pm_ = PME[a][cpsh]; pq_ = PQ9[a][cpsh]
        ax = np.abs(x)
        X = np.column_stack([
            x, ax, gm_, gs_, gq_, pm_, pq_, (x / gs_).astype(np.float32),
            (x / np.maximum(pq_, 1e-3)).astype(np.float32), (x / np.maximum(gq_, 1e-3)).astype(np.float32),
        ]).astype(np.float32)
        return X, x, y, cpsh, cgsh

    FEATS = ["x", "ax", "gmean", "gstd", "gq90", "pmean", "pq90", "x_z", "x_rel_p", "x_rel_g"]

    Xh_l = []; xh_l = []; yh_l = []; cph_l = []
    Xt_l = []; rt_l = []

    # M_ctx: 5 个 source→hepg2_all 配对; hold 行 (bench-100 靶) 只进 eval
    for a in [c for c in CTX if c != "hepg2_all"]:
        X, x, y, cpsh, _ = pair_features(a, "hepg2_all")
        hold = np.isin(cpsh, hold_codes)
        tr = ~hold
        if hold.any():
            Xh_l.append(X[hold]); xh_l.append(x[hold]); yh_l.append(y[hold]); cph_l.append(cpsh[hold])
        Xt_l.append(X[tr]); rt_l.append((y - x)[tr])
        print(f"M_ctx {a}->hepg2: train={int(tr.sum()):,} hold={int(hold.sum()):,} t={time.time()-t0:.0f}s", flush=True)
        del X, x, y, cpsh

    X_ctx = np.concatenate(Xt_l); r_ctx = np.concatenate(rt_l)
    X_hold = np.concatenate(Xh_l); x_hold = np.concatenate(xh_l)
    y_hold = np.concatenate(yh_l); cp_hold = np.concatenate(cph_l)
    del Xt_l, rt_l, Xh_l, xh_l, yh_l, cph_l
    print(f"M_ctx train {X_ctx.shape}, hold {X_hold.shape}, t={time.time()-t0:.0f}s", flush=True)

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

    # ---- 基线 (hold 上直接读数, 不外推) ----
    mae0, corr0, sp0, n0 = metrics(x_hold, y_hold, cp_hold)   # direct transfer
    mae_z = float(np.mean(np.abs(y_hold)))                     # zero baseline
    print(f"BASELINE direct: mae={mae0:.4f} corr={corr0:.4f} spearman_med={sp0:.4f} (n_pert={n0})", flush=True)
    print(f"BASELINE zero:   mae={mae_z:.4f}  (探针全池: 0.0577/0.2142/0.0976)", flush=True)

    # ---- 训练 M_ctx ----
    dtr = xgb.DMatrix(X_ctx, label=r_ctx, feature_names=FEATS)
    bst = xgb.train(params, dtr, num_boost_round=args.rounds,
                    evals=[(dtr, "train")], verbose_eval=100)
    del dtr, X_ctx, r_ctx
    bst.save_model(f"{args.out}/M_ctx.json")

    dh = xgb.DMatrix(X_hold, feature_names=FEATS)
    pr = bst.predict(dh)
    del dh, X_hold
    pred = x_hold + pr

    mae1, corr1, sp1, n1 = metrics(pred, y_hold, cp_hold)
    # 对照: 只用 x (残差=0) 已在 direct baseline; 再看纯 r̂ (诊断用)
    mae_r, corr_r, sp_r, _ = metrics(pr, y_hold - x_hold, cp_hold)

    gates = {"corr_beats_direct": corr1 > corr0,
             "mae_beats_zero": mae1 < mae_z,
             "spearman_beats_direct": sp1 > sp0}
    verdict = "PASS" if all(gates.values()) else "FAIL"
    meta = {"t_sec": round(time.time() - t0, 1), "rounds": args.rounds, "seed": args.seed,
            "params": params, "feats": FEATS,
            "n_train": int(len(pred)), "n_hold": int(len(y_hold)),
            "baseline_direct": {"mae": mae0, "corr": corr0, "spearman_med": sp0},
            "baseline_zero_mae": mae_z,
            "model_pred": {"mae": mae1, "corr": corr1, "spearman_med": sp1},
            "resid_only_diag": {"mae": mae_r, "corr": corr_r, "spearman_med": sp_r},
            "gates": gates, "verdict": verdict}
    with open(f"{args.out}/train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"MODEL pred:  mae={mae1:.4f} corr={corr1:.4f} spearman_med={sp1:.4f}", flush=True)
    print(f"RESID r̂ vs r: mae={mae_r:.4f} corr={corr_r:.4f} spearman_med={sp_r:.4f}", flush=True)
    print(f"GATES: {gates}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)

if __name__ == "__main__":
    main()
