#!/usr/bin/env python
"""C2 XGBoost 逐(靶,context)幅度头 MVP —— Bizard v2.0 (spec 冻结, REV 修正版).

规格 (勿改架构):
  样本   : data/interim/xgb_amp/features_full.parquet, 24,398 行 (6 训练 ctx × 靶)
  标签   : y = log A, A = 该(靶,ctx) top-200 |logFC| 均值 (labels.parquet 口径)
  特征   : 39 维冻结集 (REV F-1, 15 s_v1/basal + 16 se_pc + 8 fp_*) — 禁搜索
  模型   : xgboost reg:pseudohubererror, depth=7, lr=0.05, 600 rounds,
           subsample=0.8, colsample_bytree=0.8, hist, n_jobs=40, seed=0
  输出   : g = clip(exp(y_pred)/(16*sd_v1), 0.5, 3.0) 逐靶幅度乘子 (乘在 v1 Δ̂ 上)

REV 条目回应 (详见 outputs/xgb_amp/gate_report.md):
  A-1 oracle-g 逐位对拍 + 停止条件 (oracle g 全部饱和→uniform ×3)
  A-2 g 分布 + clip 饱和率报告 (>10-15% → 重参数化提案)
  B-1 E[g]=1 训练后重定心 (MVP α=0; 附 family-offset −0.30 敏感性)
  D-5 affine 校准 gate 废除 → 改 LOCO 相对判据 + 预注册校准探针定性
  F-1 特征冻结: LOCO 只做验收不做搜索

用法:
  python scripts/xgb_amplitude_head.py --stage loco      # LOCO 验收 (6 ctx)
  python scripts/xgb_amplitude_head.py --stage train     # 全量模型 → outputs/xgb_amp/
  python scripts/xgb_amplitude_head.py --stage gtables   # bench + A/B/C g 表
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
XD = ROOT / "data/interim/xgb_amp"
OUT = ROOT / "outputs/xgb_amp"
CTXS = ["gwps", "k562_ess", "rpe1", "jurkat", "hepg2", "hct116"]
# REV F-1: 39 维冻结特征集 (LOCO 之前冻结, 只验收不搜索)
FEATS = (
    ["log_sd", "log_absmean", "basal_t", "basal_t_zero", "basal_mean", "basal_nzero",
     "se_norm", "se_mu", "se_sd",
     "se_top32_en", "se_top128_en", "se_top512_en",
     "se_top32_abs", "se_top128_abs", "se_top512_abs"]
    + [f"se_pc{i}" for i in range(16)]
    + ["fp_gwps", "fp_k562_ess", "fp_rpe1", "fp_jurkat", "fp_hepg2", "fp_hct116",
       "fp_lib_size", "fp_nzero"]
)
PARAMS = dict(objective="reg:pseudohubererror", max_depth=7, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=40,
              seed=0)
ROUNDS = 600
CLIP_LO, CLIP_HI = 0.5, 3.0
LAM_BASE = 16.0  # v1 官方 λ39 ↔ 本地 lam16 anchor
# bench 4 个无覆盖靶 (s_v1+se 双缺) → g=1 回退
BENCH_FALLBACK = ["POU5F1B", "HIST1H2BM", "CYS1", "LPA"]


def load_xy() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_parquet(XD / "features_full.parquet")
    miss = [c for c in FEATS if c not in df.columns]
    if miss:
        raise KeyError(f"frozen features missing: {miss}")
    X = df[FEATS].to_numpy(np.float32)
    y = df["y"].to_numpy(np.float32)
    return df, X, y


def fit(Xtr, ytr) -> "xgb.Booster":
    return xgb.train(PARAMS, xgb.DMatrix(Xtr, label=ytr),
                     num_boost_round=ROUNDS, verbose_eval=False)


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def stage_loco(df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> None:
    """LOCO 按 ctx 轴验收: 模型 vs 两个基线 (g=1 零假设 / 训练均值常数)."""
    rows = []
    for c in CTXS:
        te = (df.ctx == c).to_numpy()
        m = fit(X[~te], y[~te])
        p = m.predict(xgb.DMatrix(X[te]))
        sd_te = df.loc[te, "sd"].to_numpy()
        base_null = np.log(16.0 * sd_te)          # g=1 即 v1 现状
        base_const = float(np.mean(y[~te]))
        rows.append(dict(
            ctx=c, n=int(te.sum()),
            rmse_model=rmse(p, y[te]),
            rmse_g1=rmse(base_null, y[te]),
            rmse_const=rmse(np.full(te.sum(), base_const), y[te]),
        ))
        print(f"[loco] {c:9s} n={rows[-1]['n']:6d} model={rows[-1]['rmse_model']:.4f} "
              f"g1={rows[-1]['rmse_g1']:.4f} const={rows[-1]['rmse_const']:.4f}")
    res = {"params": {**PARAMS, "rounds": ROUNDS}, "folds": rows,
           "pass_all_ctx": bool(all(r["rmse_model"] <= r["rmse_g1"] for r in rows))}
    (OUT / "loco_frozen.json").write_text(json.dumps(res, indent=1))
    print(f"[loco] pass_all_ctx={res['pass_all_ctx']} -> loco_frozen.json")


def stage_train(df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = fit(X, y)
    model.save_model(str(OUT / "model_final.json"))
    imp = model.get_score(importance_type="gain")
    imp_sorted = sorted(imp.items(), key=lambda kv: -kv[1])
    (OUT / "feature_importance.json").write_text(json.dumps(imp_sorted, indent=1))
    pred = model.predict(xgb.DMatrix(X))
    print(f"[train] in-sample rmse={rmse(pred, y):.4f}  trees={ROUNDS}")
    print(f"[train] top5 gain: {imp_sorted[:5]}")


def _implied_g(A: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return A / (16.0 * sd)


def _g_stats(g: np.ndarray) -> dict:
    g = np.asarray(g, float)
    return {"n": int(g.size), "mean": round(float(g.mean()), 4),
            "median": round(float(np.median(g)), 4),
            "iqr": [round(float(np.percentile(g, 25)), 4),
                    round(float(np.percentile(g, 75)), 4)],
            "sat_lo": round(float((g < CLIP_LO).mean()), 4),
            "sat_hi": round(float((g > CLIP_HI).mean()), 4)}


def _build_ctx_feats(cx: str) -> pd.DataFrame:
    """A/B/C 上下文特征 (镜像 features_full 口径): s_v1 + se600m + basal + fp 相关."""
    vocab = json.load(open(ROOT / "data/interim/l15_fusion/v35/vocab.json"))
    vmap = {g: i for i, g in enumerate(vocab)}
    sv = pd.read_parquet(ROOT / f"data/interim/xgb_amp_s_v1_{cx}.parquet")
    d = sv.reset_index()[["target_gene", "sd", "mean_abs"]].copy()
    basal = np.load(ROOT / f"data/interim/l15_fusion/v35/basal_{cx.lower()}.npy")
    tpos = d.target_gene.map(vmap)
    b_t = np.where(tpos.notna(),
                   basal[np.clip(tpos.fillna(0).astype(int), 0, len(vocab) - 1)], np.nan)
    d["basal_t"] = b_t
    d["basal_t_zero"] = (b_t == 0).astype(float)
    d["basal_mean"] = float(np.nanmean(np.abs(basal)))
    d["basal_nzero"] = float((basal > 0).mean())
    d["log_sd"] = np.log(d.sd.to_numpy())
    d["log_absmean"] = np.log(d["mean_abs"].to_numpy())
    feat = pd.read_parquet(XD / "se600m_feats.parquet")
    d = d.merge(feat, on="target_gene", how="left")
    for c in CTXS:  # fp_*: 本 ctx basal 与 6 训练 ctx basal 的 pearson (ctx 级标量)
        ob = np.load(ROOT / f"data/interim/l15_fusion/v35/basal_{c}.npy")
        d["fp_" + c] = float(np.corrcoef(basal, ob)[0, 1])
    d["fp_lib_size"] = float(np.mean(basal))
    d["fp_nzero"] = float((basal == 0).mean())
    d["ctx"] = cx
    return d


def stage_gtables(df: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = xgb.Booster(); model.load_model(str(OUT / "model_final.json"))
    report: dict = {}

    # ---- bench Q_HepG2: features_full hepg2 行 (96/100), 缺失 4 靶 g=1 ----
    bh = df[df.ctx == "hepg2"].set_index("target_gene")
    dmap = pd.read_parquet(ROOT / "data/interim/bizard/bizardnet/pred_skip_bench_hepg2.parquet")
    bench_targets = sorted(dmap.target_gene.unique())
    Xb = bh.reindex(bench_targets)[FEATS]
    covered = Xb.notna().all(axis=1).to_numpy()
    yhat = np.where(covered, model.predict(xgb.DMatrix(Xb.to_numpy(np.float32))), np.nan)
    g_raw = np.where(covered, np.clip(np.exp(yhat) / (16.0 * bh.reindex(bench_targets)["sd"].to_numpy()),
                                      CLIP_LO, CLIP_HI), 1.0)  # 含 4 缺失靶 g=1
    g_mean = float(g_raw[covered].mean())
    g_rec = g_raw / g_mean  # B-1 E[g]=1 重定心 (不再二次裁剪, 报告实际范围)
    pd.DataFrame({"target_gene": bench_targets, "g": g_raw, "g_recenter": g_rec,
                  "covered": covered}).to_parquet(OUT / "g_bench_hepg2.parquet")
    report["bench_hepg2"] = {
        "n_targets": len(bench_targets), "n_covered": int(covered.sum()),
        "fallback_g1": [t for t, c in zip(bench_targets, covered) if not c],
        "g_raw": _g_stats(g_raw[covered]), "g_recenter_mean": round(g_mean, 4),
        "g_recenter_min": round(float(g_rec.min()), 4), "g_recenter_max": round(float(g_rec.max()), 4),
    }
    print(f"[g] bench: covered={int(covered.sum())}/{len(bench_targets)} "
          f"mean_g={g_mean:.4f} sat_hi={report['bench_hepg2']['g_raw']['sat_hi']}")

    # ---- A/B/C 提交靶 g 表 + 缩放 delta ----
    for cx in ["A", "B", "C"]:
        d = _build_ctx_feats(cx)
        delta = pd.read_parquet(ROOT / f"data/interim/bizard/bizardnet/delta_submit_{cx}.parquet")
        targets = sorted(delta.target_gene.unique())
        Xi = d.set_index("target_gene").reindex(targets)
        cov = Xi.notna().all(axis=1).to_numpy() & Xi["sd"].notna().to_numpy() & (Xi["sd"].to_numpy() > 0)
        yhat = np.where(cov, model.predict(xgb.DMatrix(Xi[FEATS].to_numpy(np.float32))), np.nan)
        g_raw = np.where(cov, np.clip(np.exp(yhat) / (16.0 * Xi["sd"].to_numpy()),
                                      CLIP_LO, CLIP_HI), 1.0)
        g_mean = float(g_raw[cov].mean())
        gs = pd.Series(g_raw, index=targets)
        delta_scaled = delta.copy()
        delta_scaled["delta"] = delta_scaled.delta.to_numpy() * gs.reindex(delta_scaled.target_gene).to_numpy()
        delta_scaled.to_parquet(ROOT / f"data/interim/xgb_amp/delta_submit_g_{cx}.parquet")
        pd.DataFrame({"target_gene": targets, "g": g_raw, "g_recenter": g_raw / g_mean,
                      "covered": cov}).to_parquet(OUT / f"g_submit_{cx}.parquet")
        report[f"submit_{cx}"] = {"n_targets": len(targets), "n_covered": int(cov.sum()),
                                  "fallback_g1": [t for t, c in zip(targets, cov) if not c],
                                  "g_raw": _g_stats(g_raw[cov]), "g_mean_recenter": round(g_mean, 4)}
        print(f"[g] {cx}: covered={int(cov.sum())}/{len(targets)} mean_g={g_mean:.4f} "
              f"sat_hi={report[f'submit_{cx}']['g_raw']['sat_hi']}")

    # ---- 训练隐含 g (implied) 分布, A-2 报告 ----
    imp = _implied_g(df.A.to_numpy(), df.sd.to_numpy())
    report["train_implied_g"] = _g_stats(imp)
    report["train_implied_g_hepg2"] = _g_stats(imp[(df.ctx == "hepg2").to_numpy()])
    report["params"] = {"lam_local": LAM_BASE, "clip": [CLIP_LO, CLIP_HI]}
    (OUT / "g_report.json").write_text(json.dumps(report, indent=1))
    print("[g] wrote g_report.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["loco", "train", "gtables", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("loco", "all"):
        df, X, y = load_xy()
        stage_loco(df, X, y)
    if args.stage in ("train", "all"):
        df, X, y = load_xy()
        stage_train(df, X, y)
    if args.stage in ("gtables", "all"):
        df, _, _ = load_xy()
        stage_gtables(df)


if __name__ == "__main__":
    main()
