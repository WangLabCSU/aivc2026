#!/usr/bin/env python
"""v2.1-(c) V6: fp_self 重定义为最近邻 ctx 相关度 (预注册 spec ml_v21c_fp_loco_spec.md §3).

唯一改动 (spec 预注册, REV C1-C4 认可):
    fp'_c(t) = corr(basal_t, basal_c)              若 t != c   (与冻结版相同, 不动)
    fp'_c(t) = max_{j!=t} corr(basal_t, basal_j)   若 t =  c   (自相关 1.0 -> NN 相关度)

实现落位 = xgb_amplitude_head.py `_build_ctx_feats` 的 self 分支替换:
  冻结版: d[f"fp_{cx}"] = 1.0
  V6 版 : d[f"fp_{cx}"] = max(fp_<j>(cx) for j != cx)   # 即最近邻相关度
本脚本在数据侧直接对 features_full.parquet 的 fp 列做同一改动 (逐位等价),
其余 38 维 FEATS / PARAMS / ROUNDS / LOCO 协议与冻结版逐字相同, 冻结锚不动。

预注册判定 (先于运行锁死, spec §3):
  主判据: 六轴全部 rmse_V6 <= rmse_g1 (点估计)。
  任一轴 FAIL (预期主结果) -> (c) 终局关闭, 出负结果纪要。
  六轴全 PASS (低预期) -> seed=7 复核 + 6 模型 multi-split 方差, 再走 REV gate_report。

用法: python scripts/xgb_fp_v6_loco.py
产物: outputs/xgb_amp/fp_v6_loco.json (C1: git add -f 入档)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
XD = ROOT / "data/interim/xgb_amp"
OUT = ROOT / "outputs/xgb_amp"
CTXS = ["gwps", "k562_ess", "rpe1", "jurkat", "hepg2", "hct116"]
# 冻结 39 维 FEATS / PARAMS / ROUNDS — 逐字同 xgb_amplitude_head.py, 不改
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

# REV 独立重算过的 V6 测试侧 self 常数 (rev_v21c_fp_loco_spec_review.md 一.4) — sanity 锚
REV_NN = {"rpe1": 0.724137, "jurkat": 0.7966, "hct116": 0.6838,
          "hepg2": 0.7190, "gwps": 0.9819, "k562_ess": 0.9819}


def apply_v6(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """V6 唯一改动: ctx==t 行的 fp_<t> 列 1.0 -> NN_t = max_{j!=t} fp_<j>(t)."""
    df = df.copy()
    nn_log = {}
    for t in CTXS:
        idx = df.index[df.ctx == t]
        # 前置校验: 冻结版 self 列在自家行上恒为 ~1.0 (fp32 舍入容差内; 管线忠实前提)
        raw = df.loc[idx, f"fp_{t}"]
        assert float(np.ptp(raw.to_numpy(float))) == 0.0 and abs(float(raw.iloc[0]) - 1.0) < 1e-6, \
            f"fp_{t} on ctx=={t} rows is not constant 1.0 (got [{raw.min()},{raw.max()}])"
        # V6: NN_t = 该 ctx 行上其余 5 个 fp_<j> 的最大值 (数据侧重算, 与 _build_ctx_feats
        # self 分支逐位等价 — fp_<j>(t) 即 corr(basal_t, basal_j))
        others = df.loc[idx, [f"fp_{j}" for j in CTXS if j != t]].to_numpy(float)
        nn = float(others.max())
        df.loc[idx, f"fp_{t}"] = np.float32(nn)
        nn_log[t] = {"nn": round(nn, 6), "rev_anchor": REV_NN[t],
                     "match_rev": bool(abs(nn - REV_NN[t]) < 5e-4),
                     "n_rows": int(len(idx))}
        print(f"[v6] fp_{t}: 1.0 -> {nn:.6f} (REV anchor {REV_NN[t]}, "
              f"match={nn_log[t]['match_rev']}, rows={len(idx)})")
    return df, nn_log


def fit(Xtr, ytr) -> "xgb.Booster":
    return xgb.train(PARAMS, xgb.DMatrix(Xtr, label=ytr),
                     num_boost_round=ROUNDS, verbose_eval=False)


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> None:
    df_raw = pd.read_parquet(XD / "features_full.parquet")
    miss = [c for c in FEATS if c not in df_raw.columns]
    if miss:
        raise KeyError(f"frozen features missing: {miss}")

    # 对称性 sanity (REV 机制备注): V6 测试值 NN_t 应 = 训练侧 fp_<t> 列的 max
    sym = {}
    for t in CTXS:
        tr_rows = df_raw.index[df_raw.ctx != t]
        sym[t] = float(df_raw.loc[tr_rows, f"fp_{t}"].max())
        if abs(sym[t] - REV_NN[t]) > 5e-4:
            print(f"[v6][warn] symmetry check {t}: train-max fp_{t}={sym[t]:.6f} "
                  f"vs REV_NN {REV_NN[t]}")

    df, nn_log = apply_v6(df_raw)

    # 冻结臂对照: 同一管线在未改动矩阵上重跑, 必须逐位复现 loco_frozen.json
    X0 = df_raw[FEATS].to_numpy(np.float32)
    y = df_raw["y"].to_numpy(np.float32)
    X6 = df[FEATS].to_numpy(np.float32)

    folds, frozen_ref = [], {}
    anchor = json.load(open(OUT / "loco_frozen.json"))
    for f in anchor["folds"]:
        frozen_ref[f["ctx"]] = f["rmse_model"]

    for c in CTXS:
        te = (df.ctx == c).to_numpy()
        m6 = fit(X6[~te], y[~te])
        p6 = m6.predict(xgb.DMatrix(X6[te]))
        m0 = fit(X0[~te], y[~te])
        p0 = m0.predict(xgb.DMatrix(X0[te]))
        sd_te = df.loc[te, "sd"].to_numpy()
        base_null = np.log(16.0 * sd_te)          # g=1 即 v1 现状
        r6, r0 = rmse(p6, y[te]), rmse(p0, y[te])
        assert r0 == frozen_ref[c], \
            f"frozen arm drift on {c}: {r0!r} vs anchor {frozen_ref[c]!r}"
        folds.append(dict(
            ctx=c, n=int(te.sum()),
            rmse_v6=r6, rmse_frozen_rerun=r0, rmse_g1=rmse(base_null, y[te]),
            delta_vs_g1=r6 - rmse(base_null, y[te]),
            delta_v6_vs_frozen=r6 - r0,
        ))
        print(f"[v6][loco] {c:9s} n={folds[-1]['n']:6d} V6={r6:.4f} "
              f"frozen={r0:.4f} g1={folds[-1]['rmse_g1']:.4f} "
              f"dV6={folds[-1]['delta_vs_g1']:+.4f}")

    res = {
        "spec": "docs/team/ml_v21c_fp_loco_spec.md §3 V6",
        "rev_review": "docs/team/rev_v21c_fp_loco_spec_review.md (C1-C4)",
        "change": "fp_self: 1.0 -> max_{j!=t} corr(basal_t, basal_j); 其余 38 维/PARAMS/ROUNDS 不动",
        "nn_values": nn_log,
        "symmetry_train_max": sym,
        "params": {**PARAMS, "rounds": ROUNDS},
        "folds": folds,
        "pass_all_ctx": bool(all(f["rmse_v6"] <= f["rmse_g1"] for f in folds)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fp_v6_loco.json").write_text(json.dumps(res, indent=1))
    print(f"[v6] pass_all_ctx={res['pass_all_ctx']} -> fp_v6_loco.json")


if __name__ == "__main__":
    main()
