#!/usr/bin/env python
"""bzr1 amp v1 — L269 幅度头: per-(target,gene,context) 幅度 |y| 预测 (iter037-续2).

设计面 (ledger 冻结条款: 输出只作 HY8 幅度源替换, 永不独立成压):
  - 目标 = |y|: 只学幅度, 方向仍由 HY 构造给出 (sign(X) ∧ X*YH>0 门)
  - 双目标函数: reg:squarederror (pds/cosine 对齐 — 行内大幅度主导余弦) +
    reg:pseudohubererror (稳健对照)
  - 特征 = v2 14 列 + 幅度谱 9 列:
      rk_desc    行内 |x| 降序百分位 (1=行最大)
      wd         行内 |x| 严格更大的个数 (0=行最大)
      top8/top50 指示
      kth8/kth50 行内第8/50大 |x| (DE 括号水平)
      ax_rel_kth8
      ntc_cv / ntc_theta  NTC 变异系数 / sim8 显著性门 (2.5*cv+0.05)
诚实门 (hold-100 bench perts, 预注册):
  G1 mae(|ŷ|,|y|) < mae(|x|,|y|)
  G2 spearman(|ŷ|,|y|) med > spearman(|x|,|y|) med
  G3 mae(|ŷ|,|y|) < mae(|yh|,|y|)   (必须超 v2 校准器的幅度读数)
  G4 spearman(|ŷ|,|y|) med > spearman(|yh|,|y|) med
输出: M_amp_sq.json / M_amp_hu.json / amp_hold_v1.parquet / train_meta_amp_v1.json
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

    # ---- 词汇表 (v2 同款) ----
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
        cnt = np.bincount(inv, minlength=len(keys_u)).astype(np.float64)
        ssum = np.bincount(inv, weights=lf.astype(np.float64), minlength=len(keys_u))
        lf = (ssum / cnt).astype(np.float32)
        del keys, inv
        return keys_u, lf

    data = {}
    for n in CTX:
        data[n] = load_keys(n)
        print(f"loaded {n}: {len(data[n][0]):,} t={time.time()-t0:.0f}s", flush=True)

    # v2 校准器输出 (G3/G4 基线 + 后续 probe 的符号裁判)
    CALY = {}
    for a in SRC:
        if os.path.exists(f"{SIG}/{a}_cal.parquet"):
            CALY[a] = load_keys(f"{a}_cal")
        else:
            CALY[a] = (np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    print(f"cal refs loaded: {[a for a in CALY if len(CALY[a][0])>0]} t={time.time()-t0:.0f}s", flush=True)

    # ---- 源侧统计 (hepg2 剔 hold) + 目标侧基因先验 (v2 同款) ----
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

    # ---- NTC 幅度特征 (sim5 基因名 → vocab 对齐) ----
    g5 = pd.Index(np.load(f"{ROOT}/tmp/bzr1/sim5_genes.npy", allow_pickle=True).astype(str))
    ntc_mean = np.load(f"{ROOT}/tmp/bzr1/ntc_mean_g6146.npy").astype(np.float64)
    ntc_sd = np.load(f"{ROOT}/tmp/bzr1/ntc_sd_g6146.npy").astype(np.float64)
    cv6 = (ntc_sd / np.maximum(ntc_mean, 0.5)).astype(np.float32)
    p5 = vocab_g.get_indexer(g5)
    mm = p5 >= 0
    NTCV = np.full(NG, float(np.median(cv6)), dtype=np.float32)
    NTCV[p5[mm]] = cv6[mm]
    THETA = (2.5 * NTCV + 0.05).astype(np.float32)
    print(f"ntc aligned {int(mm.sum())}/{NG} t={time.time()-t0:.0f}s", flush=True)

    # ---- 源内幅度谱统计 (仅用 source 自身 |x| 结构, 无真值泄漏) ----
    SRC = [c for c in CTX if c != "hepg2_all"]
    SRCID = {c: i for i, c in enumerate(SRC)}
    NP_ = len(vocab_p)
    RK = {}; WD = {}; K8 = {}; K50 = {}
    for a in SRC:
        keys, lf = data[a]
        cp = (keys // NG).astype(np.int64)
        alf = np.abs(lf)
        ord_ = np.lexsort((alf, cp))
        cp_s, alf_s = cp[ord_], alf[ord_]
        starts = np.searchsorted(cp_s, np.arange(NP_), side="left")
        ends = np.searchsorted(cp_s, np.arange(NP_), side="right")
        size = np.maximum(ends - starts, 1)
        within = np.arange(len(cp_s)) - starts[cp_s]        # 升序 0-based
        rk_desc_s = (size[cp_s] - within) / size[cp_s]      # 1=最大
        rk = np.empty(len(cp), np.float32); rk[ord_] = rk_desc_s
        wd = np.empty(len(cp), np.int32); wd[ord_] = (size[cp_s] - 1 - within)
        ok8 = (ends - starts) >= 8; ok50 = (ends - starts) >= 50
        kth8 = alf_s[np.maximum(ends - 1, 0)].copy()        # <8 基因的 pert: 用最大值
        kth8[ok8] = alf_s[ends[ok8] - 8]
        kth50 = alf_s[np.maximum(ends - 1, 0)].copy()
        kth50[ok50] = alf_s[ends[ok50] - 50]
        RK[a] = rk; WD[a] = wd; K8[a] = kth8; K50[a] = kth50
        print(f"amp-rank {a} done t={time.time()-t0:.0f}s", flush=True)

    def pair_features(a):
        ka, xa = data[a]; kb, xb = data["hepg2_all"]
        shared, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
        x = xa[ia]; y = xb[ib]
        cgsh = (shared % NG).astype(np.int64); cpsh = (shared // NG).astype(np.int64)
        kc, yc = CALY[a]  # 无 cal 输出的源已在 CALY 预置空表
        pos = np.searchsorted(kc, shared)
        pos_c = np.minimum(pos, len(kc) - 1)
        yh = np.where(kc[pos_c] == shared, yc[pos_c], 0.0).astype(np.float32)
        rk = RK[a][ia]; wd = WD[a][ia]
        X = np.column_stack([
            x, np.abs(x), GME[a][cgsh], GST[a][cgsh], GQ9[a][cgsh],
            PME[a][cpsh], PQ9[a][cpsh],
            (x / GST[a][cgsh]).astype(np.float32),
            (x / np.maximum(PQ9[a][cpsh], 1e-3)).astype(np.float32),
            (x / np.maximum(GQ9[a][cgsh], 1e-3)).astype(np.float32),
            TGM[cgsh], TGSD[cgsh], TGQ9[cgsh],
            np.full(len(shared), SRCID[a], dtype=np.float32),
            rk, wd.astype(np.float32),
            (wd <= 7).astype(np.float32), (wd <= 49).astype(np.float32),
            K8[a][cpsh], K50[a][cpsh],
            (np.abs(x) / np.maximum(K8[a][cpsh], 1e-3)).astype(np.float32),
            NTCV[cgsh], THETA[cgsh],
        ]).astype(np.float32)
        return X, x, y, yh, cpsh, cgsh

    FEATS = ["x", "ax", "gmean", "gstd", "gq90", "pmean", "pq90", "x_z", "x_rel_p", "x_rel_g",
             "tg_mean", "tg_std", "tg_q90", "src_id",
             "rk_desc", "wd", "top8", "top50", "kth8", "kth50", "ax_rel_kth8",
             "ntc_cv", "ntc_theta"]

    Xh_l = []; xh_l = []; y_hold_l = []; yh_hold_l = []; cph_l = []; cgh_l = []
    Xt_l = []; at_l = []
    hold_sizes = []
    for a in SRC:
        X, x, y, yh, cpsh, cgsh = pair_features(a)
        at = np.abs(y)
        hold = np.isin(cpsh, hold_codes)
        tr = ~hold
        if hold.any():
            Xh_l.append(X[hold]); xh_l.append(x[hold])
            yh_hold_l.append(yh[hold])
            cph_l.append(cpsh[hold]); cgh_l.append(cgsh[hold])
            hold_sizes.append(int(hold.sum()))
        Xt_l.append(X[tr]); at_l.append(at[tr])
        del X, x, y, yh, cpsh, cgsh
        print(f"M_amp {a}: train={int(tr.sum()):,} hold={int(hold.sum()):,} t={time.time()-t0:.0f}s", flush=True)

    X_ctx = np.concatenate(Xt_l); a_ctx = np.concatenate(at_l)
    X_hold = np.concatenate(Xh_l); x_hold = np.concatenate(xh_l)
    # a_hold: pair_features 第二次调用重算 hold 的 |y| (只 4 源, 代价可接受)
    y_hold_list = []
    for a in SRC:
        _X, _x, _y, _yh, _cp, _cg = pair_features(a)
        _hm = np.isin(_cp, hold_codes)
        y_hold_list.append(np.abs(_y[_hm]))
    a_hold = np.concatenate(y_hold_list)
    yh_hold = np.concatenate(yh_hold_l)
    cp_hold = np.concatenate(cph_l); cg_hold = np.concatenate(cgh_l)
    assert len(a_hold) == len(X_hold) == sum(hold_sizes)
    print(f"train {X_ctx.shape}, hold {X_hold.shape}, t={time.time()-t0:.0f}s", flush=True)

    def metrics(pred, y, cp):
        mae = float(np.mean(np.abs(pred - y)))
        corr = float(np.corrcoef(pred, y)[0, 1])
        sps = []
        for c in np.unique(cp):
            g = cp == c
            if g.sum() >= 30 and np.std(pred[g]) > 0 and np.std(y[g]) > 0:
                sps.append(spearmanr(pred[g], y[g]).statistic)
        return mae, corr, float(np.median(sps))

    ax_hold = np.abs(x_hold)
    ayh_hold = np.abs(yh_hold)
    mae_x, corr_x, sp_x = metrics(ax_hold, a_hold, cp_hold)
    mae_yh, corr_yh, sp_yh = metrics(ayh_hold, a_hold, cp_hold)
    gp = np.abs(TGM[cg_hold])
    mae_g, corr_g, sp_g = metrics(gp, a_hold, cp_hold)
    print(f"BASELINE |x|:        mae={mae_x:.4f} corr={corr_x:.4f} spearman_med={sp_x:.4f}", flush=True)
    print(f"BASELINE |yh| (v2):  mae={mae_yh:.4f} corr={corr_yh:.4f} spearman_med={sp_yh:.4f}", flush=True)
    print(f"BASELINE gene-prior: mae={mae_g:.4f} corr={corr_g:.4f} spearman_med={sp_g:.4f}", flush=True)

    preds = {}
    for tag, pobj in (("sq", {"objective": "reg:squarederror"}),
                      ("hu", {"objective": "reg:pseudohubererror", "huber_slope": 0.05})):
        params = {"tree_method": "hist", "device": "cuda", "max_depth": 8, "eta": 0.05,
                  "subsample": 0.7, "colsample_bytree": 0.6, "seed": args.seed, **pobj}
        dtr = xgb.DMatrix(X_ctx, label=a_ctx, feature_names=FEATS)
        bst = xgb.train(params, dtr, num_boost_round=args.rounds,
                        evals=[(dtr, "train")], verbose_eval=100)
        del dtr
        bst.save_model(f"{args.out}/M_amp_{tag}.json")
        dh = xgb.DMatrix(X_hold, feature_names=FEATS)
        preds[tag] = bst.predict(dh)
        del dh
        print(f"trained amp-{tag} t={time.time()-t0:.0f}s", flush=True)

    # hold 明细落盘 (probe 用): 每 source 的 hold 对 (p,g) + 幅度预测
    parts = []
    off = 0
    for a, hs in zip(SRC, hold_sizes):
        ka, _xa = data[a]
        shared, _ia, _ib = np.intersect1d(ka, data["hepg2_all"][0],
                                          assume_unique=True, return_indices=True)
        cph = (shared // NG); cgh = (shared % NG)
        hm = np.isin(cph, hold_codes)
        assert int(hm.sum()) == hs
        df_a = pd.DataFrame({
            "src": a,
            "target_gene": vocab_p[cph[hm]].astype(str),
            "gene": vocab_g[cgh[hm]].astype(str),
            "abs_sq": preds["sq"][off:off + hs],
            "abs_hu": preds["hu"][off:off + hs],
        })
        parts.append(df_a)
        off += hs
    dump = pd.concat(parts, ignore_index=True)
    dump.to_parquet(f"{args.out}/amp_hold_v1.parquet", index=False)
    print(f"hold dump {dump.shape} -> amp_hold_v1.parquet t={time.time()-t0:.0f}s", flush=True)

    meta = {"t_sec": round(time.time() - t0, 1), "rounds": args.rounds, "seed": args.seed,
            "feats": FEATS, "formulation": "amp |y| dual-objective (sq+huber)",
            "n_train": int(len(a_ctx)), "n_hold": int(len(a_hold)),
            "baseline_abs_x": {"mae": mae_x, "corr": corr_x, "spearman_med": sp_x},
            "baseline_abs_yh": {"mae": mae_yh, "corr": corr_yh, "spearman_med": sp_yh},
            "baseline_gene_prior_abs": {"mae": mae_g, "corr": corr_g, "spearman_med": sp_g}}
    for tag in ("sq", "hu"):
        mae1, corr1, sp1 = metrics(preds[tag], a_hold, cp_hold)
        gates = {"G1_mae_beats_x": mae1 < mae_x,
                 "G2_sp_beats_x": sp1 > sp_x,
                 "G3_mae_beats_yh": mae1 < mae_yh,
                 "G4_sp_beats_yh": sp1 > sp_yh}
        verdict = "PASS" if all(gates.values()) else "FAIL"
        meta[f"model_{tag}"] = {"mae": mae1, "corr": corr1, "spearman_med": sp1,
                                "gates": gates, "verdict": verdict}
        print(f"MODEL amp-{tag}: mae={mae1:.4f} corr={corr1:.4f} spearman_med={sp1:.4f} "
              f"GATES={'ALL PASS' if verdict == 'PASS' else gates}", flush=True)
    with open(f"{args.out}/train_meta_amp_v1.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"[done] t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
