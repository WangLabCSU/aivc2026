#!/usr/bin/env python
"""bzr1 v2 校准器推理: 源签名 → hepg2-like 校准签名落盘 (v0.8 推理路径, iter030-修正4).

对官方链 Δ 源 (gwps, hct116_all) 每 (target,gene) 行构建 v2 同款 14 特征,
用 M_ctx_v2.json 预测 ŷ (hepg2-like 响应), 以 {target_gene, gene, logfc=ŷ} 落盘
data/interim/signatures/{src}_cal.parquet.
官方链零代码改动: config datasets 换 [gwps_cal, hct116_all_cal] 即复用 make_predictions.load_delta.

诚实性: tg 先验严格 hepg2 train-perts (bench-100 hold 全场景剔除, 与 v2 训练一致);
源侧统计全量 (源侧无 hold 概念, 与链上 raw 签名可见性一致).
词汇表只需本运行内自洽 (3 源并集), 无须与训练 vocab 逐点一致 — 模型只见特征不看码.
"""
import argparse, os, time
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
VOCAB_SRC = ["gwps", "hct116_all", "hepg2_all"]  # 官方链两源 + 目标侧先验源
SRC = ["gwps", "hct116_all"]
SRCID = {c: i for i, c in enumerate(SRC)}
FEATS = ["x", "ax", "gmean", "gstd", "gq90", "pmean", "pq90", "x_z", "x_rel_p", "x_rel_g",
         "tg_mean", "tg_std", "tg_q90", "src_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{ROOT}/tmp/bzr1/M_ctx_v2.json")
    ap.add_argument("--out-dir", default=SIG)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 词汇表 (3 源并集; 仅需本运行内自洽) ----
    genes, perts = set(), set()
    for n in VOCAB_SRC:
        df = pd.read_parquet(f"{SIG}/{n}.parquet", columns=["target_gene", "gene"])
        genes |= set(pd.unique(df["gene"].astype(str)))
        perts |= set(pd.unique(df["target_gene"].astype(str)))
        del df
    vocab_g = pd.Index(sorted(genes)); vocab_p = pd.Index(sorted(perts))
    NG = len(vocab_g)
    gp = vocab_p.to_numpy(); gg = vocab_g.to_numpy()
    del genes, perts
    print(f"vocab: g={NG} p={len(vocab_p)} t={time.time()-t0:.0f}s", flush=True)

    bench = pd.read_csv(f"{ROOT}/outputs/metrics/local_bench/dualdeep_lam70mf14_Q/Q_HepG2_eval/results.csv")
    hold_codes = np.where(vocab_p.isin(set(bench["perturbation"].astype(str))))[0]
    assert len(hold_codes) == 100, f"expect 100 hold perts, got {len(hold_codes)}"

    def load_keys(name):
        df = pd.read_parquet(f"{SIG}/{name}.parquet", columns=["target_gene", "gene", "logfc"])
        cp = vocab_p.get_indexer(df["target_gene"].astype(str).values)
        cg = vocab_g.get_indexer(df["gene"].astype(str).values)
        assert (cp >= 0).all() and (cg >= 0).all(), f"{name}: unseen keys?!"
        keys = (cp.astype(np.int64) * NG + cg.astype(np.int64))
        lf = df["logfc"].values.astype(np.float32)
        del df
        keys_u, inv = np.unique(keys, return_inverse=True)
        # 对齐修复同 iter030-修正3: 值无条件经 inv 跟随重排
        cnt = np.bincount(inv, minlength=len(keys_u)).astype(np.float64)
        ssum = np.bincount(inv, weights=lf.astype(np.float64), minlength=len(keys_u))
        lf = (ssum / cnt).astype(np.float32)
        del keys, inv
        return keys_u, lf

    def to_global(stat_series, vocab, default):
        arr = np.full(len(vocab), default, dtype=np.float32)
        arr[np.asarray(stat_series.index, dtype=np.int64)] = stat_series.values.astype(np.float32)
        return arr

    # ---- hepg2 目标侧基因先验 (train perts only; 与 v2 训练逐行一致) ----
    kh, yh = load_keys("hepg2_all")
    cph = (kh // NG).astype(np.int64); cgh = (kh % NG).astype(np.int64)
    m_tr = ~np.isin(cph, hold_codes)
    dfg = pd.DataFrame({"cg": cgh[m_tr], "lf": yh[m_tr], "alf": np.abs(yh[m_tr])})
    tg = dfg.groupby("cg")["lf"]
    TGM = to_global(tg.mean(), vocab_g, 0.0)
    TGSD = to_global(tg.std().fillna(1.0), vocab_g, 0.09)
    TGQ9 = to_global(dfg.groupby("cg")["alf"].quantile(0.9), vocab_g, 0.05)
    del dfg, kh, yh
    print(f"hepg2 priors done t={time.time()-t0:.0f}s", flush=True)

    bst = xgb.Booster()
    bst.load_model(args.model)

    for a in SRC:
        keys, lf = load_keys(a)
        cp = (keys // NG).astype(np.int64); cg = (keys % NG).astype(np.int64)
        alf = np.abs(lf)
        df = pd.DataFrame({"cp": cp, "cg": cg, "lf": lf, "alf": alf})
        gb = df.groupby("cg")["lf"]
        GME = to_global(gb.mean(), vocab_g, 0.0)
        GST = to_global(gb.std().fillna(1.0), vocab_g, 1.0)
        GQ9 = to_global(df.groupby("cg")["alf"].quantile(0.9), vocab_g, 0.05)
        PME = to_global(df.groupby("cp")["alf"].mean(), vocab_p, 0.05)
        PQ9 = to_global(df.groupby("cp")["alf"].quantile(0.9), vocab_p, 0.05)
        del df
        X = np.column_stack([
            lf, alf, GME[cg], GST[cg], GQ9[cg],
            PME[cp], PQ9[cp],
            # 分母保护: std=0 的基因 (重复值) 会产生 inf → XGBoost 崩; 训练配对集中无此
            # 情形, 全量推理时必须夹底 (与 x_rel_p/x_rel_g 同式)
            (lf / np.maximum(GST[cg], 1e-3)).astype(np.float32),
            (lf / np.maximum(PQ9[cp], 1e-3)).astype(np.float32),
            (lf / np.maximum(GQ9[cg], 1e-3)).astype(np.float32),
            TGM[cg], TGSD[cg], TGQ9[cg],
            np.full(len(keys), SRCID[a], dtype=np.float32),
        ]).astype(np.float32)
        del GME, GST, GQ9, PME, PQ9
        dm = xgb.DMatrix(X, feature_names=FEATS)
        del X
        yhat = bst.predict(dm)
        del dm
        qx = np.quantile(lf, [0.5, 0.9, 0.99]); qy = np.quantile(yhat, [0.5, 0.9, 0.99])
        cx = float(np.corrcoef(yhat, lf)[0, 1])
        print(f"[{a}] n={len(keys):,} x q50/90/99={np.round(qx,3)} yhat q50/90/99={np.round(qy,3)} "
              f"corr(yhat,x)={cx:.3f} t={time.time()-t0:.0f}s", flush=True)
        out = pd.DataFrame({"target_gene": gp[cp], "gene": gg[cg],
                            "logfc": yhat.astype(np.float32)})
        out.to_parquet(f"{args.out_dir}/{a}_cal.parquet", index=False)
        print(f"[{a}] wrote {args.out_dir}/{a}_cal.parquet t={time.time()-t0:.0f}s", flush=True)
        del keys, lf, yhat, out

    print(f"done t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
