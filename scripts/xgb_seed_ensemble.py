#!/usr/bin/env python
"""v2.1-(a)-v2 seed-ensemble 评测 —— K-seed 均值化 LOCO (spec: docs/team/ml_v21av2_seedensemble_spec.md).

设计冻结 (2026-10-14, REV E1-E4 逐条映射; REV 认可前不运行; 运行前不改):
  Arm A (豁免方差基建): 6 轴 × 6 split_seed {101,202,303,404,505,606} × K=5 成员
    = 180 训练. 成员共享同一 keep 索引 (与 (a) 同协议: default_rng(split_seed) 按
    训练侧 5 ctx 分层等比抽 90%, 行级 (target,ctx) 抽样), 仅 xgb seed 不同
    → 折内差异纯为 seed 分量.
  Arm B (点估计+不确定性): 全量训练 LOCO 6 轴 × 2 seed 集 = 60 训练.
    SET_A {0,7,13,42,2026} 主判据; SET_B {1,2,3,4,5} 仅作 ensemble 级
    seed-set 位移量级 (n=2 差值, 不作方向证据).
  E2 定义: ensemble = 逐样本预测算术均值 (log 空间) 后算 rmse;
    gate 预测器自单模型变为 seed-ensemble, 未来 gate_report 须逐字绑定本定义.
  E3 判定 (不放宽): 逐 FAIL 轴 |delta_ens(Arm B SET_A)| < 2×σ_split_ens(Arm A, ddof=1);
    rpe1 单轴即否决. ensemble 用自身 delta_ens (旧 delta_frozen 仅作对照备案).

用法:
  python scripts/xgb_seed_ensemble.py              # 240 训练全跑 (~35 分钟, 后台)
  python scripts/xgb_seed_ensemble.py --axis rpe1  # 单轴 (调试/续跑)
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
FRAC = 0.90
# E1 冻结清单 (spec §1 表; 运行中不加 K / 不换 seed / 不加 split)
SET_A = [0, 7, 13, 42, 2026]
SET_B = [1, 2, 3, 4, 5]
SPLIT_SEEDS = [101, 202, 303, 404, 505, 606]
FAIL_AXES = ["rpe1", "jurkat", "hct116"]
# 冻结单模型对照值 (loco_frozen.json; 仅入档对照, 判定用 ensemble 自身 delta_ens)
FROZEN_DELTA = {"gwps": -0.3518, "k562_ess": -0.0417, "hepg2": -0.0861,
                "rpe1": 0.1177, "jurkat": 0.0794, "hct116": 0.0828}


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def train_pred(seed: int, Xtr, ytr, Xte) -> np.ndarray:
    m = xgb.train({**PARAMS, "seed": seed}, xgb.DMatrix(Xtr, label=ytr),
                  num_boost_round=ROUNDS, verbose_eval=False)
    return m.predict(xgb.DMatrix(Xte))


def run_members(seeds, Xtr, ytr, Xte, yte):
    """K 个成员共享训练数据仅 seed 异; 返回逐样本均值预测 + 成员 rmse 列表."""
    preds, member_rmses = [], []
    for xs in seeds:
        p = train_pred(xs, Xtr, ytr, Xte)
        preds.append(p)
        member_rmses.append(round(rmse(p, yte), 4))
    ens = np.mean(preds, axis=0)  # E2: 逐样本算术均值 (log 空间)
    return ens, member_rmses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="all", choices=CTXS + ["all"])
    args = ap.parse_args()
    axes = CTXS if args.axis == "all" else [args.axis]

    df = pd.read_parquet(XD / "features_full.parquet")
    miss = [c for c in FEATS if c not in df.columns]
    if miss:
        raise KeyError(f"frozen features missing: {miss}")
    X = df[FEATS].to_numpy(np.float32)
    y = df["y"].to_numpy(np.float32)
    ctx_arr = df["ctx"].to_numpy()
    idx_by_ctx = {c: np.where(ctx_arr == c)[0] for c in CTXS}

    out_path = OUT / "loco_seed_ensemble.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {"folds": [], "axes": {}}
    done = {(f["ctx"], f["arm"], f.get("split_seed"), f.get("seed_set")) for f in res["folds"]}

    for c in axes:
        te = idx_by_ctx[c]
        tr_pool = np.concatenate([idx_by_ctx[k] for k in CTXS if k != c])
        yte = y[te]
        base_null = np.log(16.0 * df.loc[te, "sd"].to_numpy())
        rmse_g1 = rmse(base_null, yte)

        # ---- Arm A: 6 split × K=5 成员 (共享 keep 索引) ----
        for ss in SPLIT_SEEDS:
            key = (c, "sub", ss, None)
            if key in done:
                continue
            rng = np.random.default_rng(ss)
            # 与 (a) 完全同协议: 每 ctx 独立抽 90% (行级, 不做 target 分组)
            keep_parts = []
            for k in CTXS:
                if k == c:
                    continue
                rows = idx_by_ctx[k]
                n_keep = int(round(len(rows) * FRAC))
                keep_parts.append(rng.choice(rows, size=n_keep, replace=False))
            keep = np.concatenate(keep_parts)
            ens, mrms = run_members(SET_A, X[keep], y[keep], X[te], yte)
            row = dict(ctx=c, arm="sub", split_seed=ss, member_seeds=SET_A,
                       member_rmses=mrms,
                       member_mean=round(float(np.mean(mrms)), 4),
                       rmse_ens=round(rmse(ens, yte), 4),
                       n_train=int(len(keep)), rmse_g1=round(rmse_g1, 4))
            row["delta_ens"] = round(row["rmse_ens"] - row["rmse_g1"], 4)
            res["folds"].append(row)
            print(f"[se] {c:9s} split{ss} ens={row['rmse_ens']:.4f} "
                  f"members={mrms} g1={row['rmse_g1']:.4f}", flush=True)
            out_path.write_text(json.dumps(res, indent=1))

        # ---- Arm B: 全量训练 × {SET_A, SET_B} ----
        for set_name, seeds in [("SET_A", SET_A), ("SET_B", SET_B)]:
            key = (c, "full", 0, set_name)
            if key in done:
                continue
            ens, mrms = run_members(seeds, X[tr_pool], y[tr_pool], X[te], yte)
            row = dict(ctx=c, arm="full", seed_set=set_name, member_seeds=seeds,
                       member_rmses=mrms,
                       member_mean=round(float(np.mean(mrms)), 4),
                       rmse_ens=round(rmse(ens, yte), 4),
                       n_train=int(len(tr_pool)), rmse_g1=round(rmse_g1, 4))
            row["delta_ens"] = round(row["rmse_ens"] - row["rmse_g1"], 4)
            res["folds"].append(row)
            print(f"[se] {c:9s} FULL {set_name} ens={row['rmse_ens']:.4f} "
                  f"members={mrms}", flush=True)
            out_path.write_text(json.dumps(res, indent=1))

    # ---- 汇总: σ_split_ens + E3 判定 ----
    frozen = json.loads((OUT / "loco_frozen.json").read_text())
    frozen_rmse = {f["ctx"]: f["rmse_model"] for f in frozen["folds"]}
    frozen_g1 = {f["ctx"]: f["rmse_g1"] for f in frozen["folds"]}
    axes_out = {}
    for c in CTXS:
        subs = [f for f in res["folds"] if f["arm"] == "sub" and f["ctx"] == c]
        fulls = {f["seed_set"]: f for f in res["folds"] if f["arm"] == "full" and f["ctx"] == c}
        if len(subs) < len(SPLIT_SEEDS) or "SET_A" not in fulls:
            axes_out[c] = {"status": "incomplete",
                           "n_sub": len(subs),
                           "n_full": len(fulls)}
            continue
        ens_rmses = [f["rmse_ens"] for f in subs]
        sigma_split_ens = float(np.std(ens_rmses, ddof=1))  # C1 同款 ddof=1
        fa = fulls["SET_A"]
        delta_ens = fa["rmse_ens"] - fa["rmse_g1"]
        thr = 2.0 * sigma_split_ens
        axes_out[c] = {
            "sigma_split_ens": round(sigma_split_ens, 5),
            "ens_rmses_per_split": ens_rmses,
            "armB_SETA": fa, "armB_SETB": fulls.get("SET_B"),
            "delta_ens": round(delta_ens, 4),
            "threshold_2x_sigma_split_ens": round(thr, 5),
            "exemption_eligible": bool(abs(delta_ens) < thr),
            "delta_frozen_reference": FROZEN_DELTA[c],  # 对照备案, 不参与判定
            "frozen_rmse": frozen_rmse[c],
        }
        if "SET_B" in fulls:
            axes_out[c]["seedset_displacement_SETA_vs_SETB"] = round(
                fulls["SET_A"]["rmse_ens"] - fulls["SET_B"]["rmse_ens"], 5)
        # C2 同款逐折记录已在 folds: 5 成员各自 rmse 供 REV 独立重算
    res["axes"] = axes_out
    res["params"] = {**PARAMS, "rounds": ROUNDS, "frac": FRAC,
                     "SET_A": SET_A, "SET_B": SET_B, "split_seeds": SPLIT_SEEDS,
                     "ensemble_def": "per-sample arithmetic mean of K=5 member predictions (log space)"}
    out_path.write_text(json.dumps(res, indent=1))
    for c, a in axes_out.items():
        if a.get("status") == "incomplete":
            print(f"[sum] {c}: incomplete (sub {a['n_sub']}/6, full {a['n_full']}/2)")
        else:
            print(f"[sum] {c}: sigma_split_ens={a['sigma_split_ens']:.5f} "
                  f"delta_ens={a['delta_ens']:+.4f} thr={a['threshold_2x_sigma_split_ens']:.5f} "
                  f"eligible={a['exemption_eligible']}"
                  + (f" setdisp={a['seedset_displacement_SETA_vs_SETB']:+.5f}"
                     if "seedset_displacement_SETA_vs_SETB" in a else ""))
    rpe1 = axes_out.get("rpe1", {})
    if rpe1.get("exemption_eligible") is False:
        print("[gate] rpe1 单轴否决: 豁免通道不可用 (R-1 任一轴劣化→不发, 结构不变)")
    print("[se] done ->", out_path)


if __name__ == "__main__":
    main()
