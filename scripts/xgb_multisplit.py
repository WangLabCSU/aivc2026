#!/usr/bin/env python
"""v2.1-(a) multi-split 方差基建 —— LOCO 训练侧重采样方差 (spec: docs/team/ml_v21a_multisplit_spec.md).

设计冻结 (2026-10-14, 待 REV 认可后运行; 运行前不改):
  测试集 : 与 loco_frozen.json 完全一致 (整 ctx 留出, 同 FEATS/PARAMS/ROUNDS/CLIP)
  每轴 7 模型:
    - 3 个 90% 训练子采样 (按训练侧 5 ctx 分层等比, split_seed ∈ {101,202,303})
      × 2 个 xgb seed {0,7} = 6 个 subsample 模型
    - 1 个全量训练对照 (全 5 训练 ctx, seed=7) —— 与已入档 seed=0 全量差 = 纯种子噪声
  抽样单元 = (target, ctx) 行级随机 (spec 1ac598d: 不做 target 级分组)
  轴方差  : std_axis = std(rmse_model) 该轴 6 subsample 模型样本 std
  豁免判定: 冻结折叠点估计劣化 delta_frozen < 2×std_axis (冻结锚不换, 新运行只供方差)
  预登记  : rpe1 劣化 +0.1177 需 std>0.0589 / jurkat +0.0794 需 >0.0397 / hct116 +0.0828 需 >0.0414

用法:
  python scripts/xgb_multisplit.py            # 42 折全跑 (~数小时, 后台)
  python scripts/xgb_multisplit.py --axis rpe1  # 单轴 (调试/续跑)
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
SPLIT_SEEDS = [101, 202, 303]
XGB_SEEDS = [0, 7]
FRAC = 0.90
# 预登记三轴劣化预算 (spec 第 2 节, 防事后解读漂移)
PREREG_BUDGET = {"rpe1": 0.1177, "jurkat": 0.0794, "hct116": 0.0828}
# 冻结锚 (loco_frozen.json, commit abcd991) —— 只读取用于豁免判定, 不重跑替换
FROZEN_DELTA = {"gwps": -0.3518, "k562_ess": -0.0417, "hepg2": -0.0861,
                "rpe1": 0.1177, "jurkat": 0.0794, "hct116": 0.0828}


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


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

    out_path = OUT / "loco_multisplit.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {"folds": [], "axes": {}}
    done = {(f["ctx"], f["split_seed"], f["xgb_seed"], f["kind"]) for f in res["folds"]}

    for c in axes:
        te = idx_by_ctx[c]
        tr_pool = np.concatenate([idx_by_ctx[k] for k in CTXS if k != c])
        yte = y[te]
        base_null = np.log(16.0 * df.loc[te, "sd"].to_numpy())
        rmse_g1 = rmse(base_null, yte)
        axis_rows = []
        for ss in SPLIT_SEEDS:
            rng = np.random.default_rng(ss)
            # 分层等比: 每个训练 ctx 内独立抽 90% (行级, 不做 target 分组)
            keep_parts = []
            for k in CTXS:
                if k == c:
                    continue
                rows = idx_by_ctx[k]
                n_keep = int(round(len(rows) * FRAC))
                keep_parts.append(rng.choice(rows, size=n_keep, replace=False))
            keep = np.concatenate(keep_parts)
            for xs in XGB_SEEDS:
                key = (c, ss, xs, "sub")
                if key in done:
                    continue
                params = {**PARAMS, "seed": xs}
                m = xgb.train(params, xgb.DMatrix(X[keep], label=y[keep]),
                              num_boost_round=ROUNDS, verbose_eval=False)
                p = m.predict(xgb.DMatrix(X[te]))
                row = dict(ctx=c, kind="sub", split_seed=ss, xgb_seed=xs, n_train=int(len(keep)),
                           rmse_model=round(rmse(p, yte), 4), rmse_g1=round(rmse_g1, 4))
                row["delta"] = round(row["rmse_model"] - row["rmse_g1"], 4)  # C2 逐折 delta
                res["folds"].append(row)
                axis_rows.append(row)
                print(f"[ms] {c:9s} split{ss} seed{xs} rmse={row['rmse_model']:.4f} "
                      f"g1={row['rmse_g1']:.4f}", flush=True)
        # 全量 seed=7 对照 (每轴同一个数, 只算一次)
        key = (c, 0, 7, "full")
        if key not in done:
            tr = np.concatenate([idx_by_ctx[k] for k in CTXS if k != c])
            m = xgb.train({**PARAMS, "seed": 7}, xgb.DMatrix(X[tr], label=y[tr]),
                          num_boost_round=ROUNDS, verbose_eval=False)
            p = m.predict(xgb.DMatrix(X[te]))
            row = dict(ctx=c, kind="full", split_seed=0, xgb_seed=7, n_train=int(len(tr)),
                       rmse_model=round(rmse(p, yte), 4), rmse_g1=round(rmse_g1, 4))
            row["delta"] = round(row["rmse_model"] - row["rmse_g1"], 4)
            res["folds"].append(row)
            print(f"[ms] {c:9s} FULL seed7 rmse={row['rmse_model']:.4f}", flush=True)
        (OUT).mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=1))

    # ---- 汇总: std_axis + 豁免判定 (全部轴跑完后或每轴增量) ----
    sub = [f for f in res["folds"] if f["kind"] == "sub"]
    full = [f for f in res["folds"] if f["kind"] == "full"]
    frozen = json.loads((OUT / "loco_frozen.json").read_text())
    frozen_rmse = {f["ctx"]: f["rmse_model"] for f in frozen["folds"]}
    frozen_g1 = {f["ctx"]: f["rmse_g1"] for f in frozen["folds"]}
    axes_out = {}
    for c in CTXS:
        rs = [f["rmse_model"] for f in sub if f["ctx"] == c]
        if len(rs) < 6:
            axes_out[c] = {"status": "incomplete", "n_sub": len(rs)}
            continue
        std_axis = float(np.std(rs, ddof=1))
        delta_frozen = frozen_rmse[c] - frozen_g1[c]
        need = 2.0 * std_axis
        axes_out[c] = {
            "rmse_models": rs, "mean": round(float(np.mean(rs)), 4),
            "std_axis": round(std_axis, 5),
            "frozen_rmse": frozen_rmse[c], "delta_frozen": round(delta_frozen, 4),
            "threshold_2std": round(need, 5),
            "exemption_eligible": bool(abs(delta_frozen) < need),
            "prereg_budget": PREREG_BUDGET.get(c),
        }
        fl = [f for f in full if f["ctx"] == c]
        if fl:
            axes_out[c]["seed_noise_full_s7_vs_frozen_s0"] = round(
                fl[0]["rmse_model"] - frozen_rmse[c], 5)
        # C2 分解报告: 每 split_seed 均值 + seed 对内差, 供 REV 独立重算 split-only 口径
        pairs = [f for f in res["folds"] if f["kind"] == "sub" and f["ctx"] == c]
        by_split = {}
        for f in pairs:
            by_split.setdefault(f["split_seed"], []).append(f["rmse_model"])
        axes_out[c]["decomp"] = {
            f"split{ss}": {"mean": round(float(np.mean(v)), 4),
                           "seed_pair_gap": round(abs(v[0] - v[1]), 4)}
            for ss, v in sorted(by_split.items())}
    res["axes"] = axes_out
    res["params"] = {**PARAMS, "rounds": ROUNDS, "frac": FRAC,
                     "split_seeds": SPLIT_SEEDS, "xgb_seeds": XGB_SEEDS}
    out_path.write_text(json.dumps(res, indent=1))
    for c, a in axes_out.items():
        if a.get("status") == "incomplete":
            print(f"[sum] {c}: incomplete ({a['n_sub']}/6)")
        else:
            print(f"[sum] {c}: std={a['std_axis']:.4f} delta_frozen={a['delta_frozen']:+.4f} "
                  f"2xstd={a['threshold_2xstd'] if 'threshold_2xstd' in a else a['threshold_2std']:.4f} "
                  f"eligible={a['exemption_eligible']}")
    print("[ms] done ->", out_path)


if __name__ == "__main__":
    main()
