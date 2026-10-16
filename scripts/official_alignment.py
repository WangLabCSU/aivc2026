"""official-alignment bench 对齐层 v1 —— 让本地评测"说官网的语言"。

背景: 本地 anchor 归一化分数与官方分数轴不可比（bench fid 0.844 ↔ 官方 -0.034），
且排行榜揭示真正的分差来自 pds/mse/reach/nmae 的 RAW 值（大而准的逐扰动效应）。

功能:
  raw      —— 从本地 eval 的 results.csv 抽取 RAW per-metric（与排行榜 raw 列同口径对比）
  calib    —— 用台账官方点拟合 per-metric × per-family 仿射映射（official = a·local_raw + b）
  predict  —— 对任一本地候选输出"预测官方 per-metric + overall"（提交 gate 的新判据）

用法:
  python scripts/official_alignment.py raw --tag ensA050_lam65
  python scripts/official_alignment.py calib           # 拟合并落 docs/research/official_alignment_v1.json
  python scripts/official_alignment.py predict --tag ensA050_lam65 --family heur
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
OUT = ROOT / "outputs/metrics/local_bench"
CALIB = ROOT / "docs/research/official_alignment_v1.json"

# 排行榜/官方 raw 指标名（cell-eval2 名 → 排行榜短名）
METRICS = {
    "pds_cosine": "pds",
    "expr_mse_unbiased_capped": "mse",
    "de_wilcoxon_sig_jaccard": "jac",
    "de_wilcoxon_lfc_nmae": "nmae",
    "de_wilcoxon_direction_fidelity_yield_raw": "fid",
    "de_wilcoxon_direction_reach_raw": "reach",
}

# 官方点（来自 submission_ledger / *_online.json）: (family, stem, local_tag, per-metric official raw)
OFFICIAL_POINTS = {
    ("heur", "20261005_dualsrcdeep_lam65"): dict(
        local="dualdeep_lam65", official={"pds": 0.5022, "mse": 0.0, "nmae": -0.0656,
                                          "fid": -0.0340, "reach": 0.1267, "jac": 0.0023}),
    ("heur", "20261003_dualsrcdeep_lam55"): dict(
        local="dualdeep_lam55_Q_HepG2", official={"pds": 0.4769, "mse": 0.0, "nmae": -0.0079,
                                                  "fid": -0.0634, "reach": 0.1185, "jac": -0.0007}),
    ("heur", "20261003_dualsrcdeep_lam45"): dict(
        local="dualdeep_lam45_Q_HepG2", official={"pds": 0.4367, "mse": 0.0, "nmae": 0.0285,
                                                  "fid": -0.1076, "reach": 0.1060, "jac": -0.0055}),
    ("fusion", "20261004_l15fusion_v35_lam45"): dict(
        local="l15v35s2_fused_lam4.5", official={"pds": 0.3821, "mse": 0.0, "nmae": -0.0016,
                                                 "fid": -0.1768, "reach": 0.0779, "jac": -0.0121}),
    ("fusion", "20261004_l15fusion_v35_lam55"): dict(
        local="l15v35s2_fused_lam5.5", official={"pds": 0.4167, "mse": 0.0, "nmae": -0.0427,
                                                 "fid": -0.1320, "reach": 0.0838, "jac": -0.0089}),
    ("ens", "20261006_ens_alpha05_lam65"): dict(
        local="ensA050_lam65", official={"pds": 0.4839, "mse": 0.0, "nmae": -0.0607,
                                         "fid": -0.0530, "reach": 0.1112, "jac": -0.0013}),
    ("bizard", "20261008_bizard_v10"): dict(
        local="bzbiz10skip_lam16", official={"pds": 0.2852, "mse": 0.0, "nmae": -0.0168,
                                             "fid": -0.0319, "reach": 0.0711, "jac": -0.0034}),
    ("bizard", "20261012_bizard_v10_lam55"): dict(
        local="bzbiz10skip_lam55", official={"pds": 0.3045, "mse": 0.0, "nmae": -0.1743,
                                             "fid": -0.0141, "reach": 0.0815, "jac": 0.0003}),
    ("bizard", "20261014_bizard_v02_lam47"): dict(  # bizard-v0.2 entry BWy91Q4UXdusW6QcxjCU submitted 2026-10-14T12:00:27.712661Z Rank 908 Overall 0.045655
        local="bzbiz10skip_lam47", official={"pds": 0.2956, "mse": 0.0, "nmae": -0.0798,
                                             "fid": -0.0182, "reach": 0.0774, "jac": -0.0011}),
    ("bizard", "20261015_bizard_v03_lam37"): dict(  # bizard-v0.3 entry SgXJeIQwv7taWP4B0GND submitted 2026-10-15T03:52:47.721537Z Rank 886 Overall 0.050214
        local="bzbiz10skip_lam37", official={"pds": 0.2816, "mse": 0.0, "nmae": -0.0069,
                                             "fid": -0.0376, "reach": 0.0684, "jac": -0.0042}),
    ("bizard", "20261015_bizard_v04_lam35"): dict(  # bizard-v0.4 entry 58kzxoM9gGrF1QOdXS5k submitted 2026-10-15T04:22:34.450657Z Rank 887 Overall 0.049122
        local="bzbiz10skip_lam35", official={"pds": 0.2773, "mse": 0.0, "nmae": 0.0020,
                                             "fid": -0.0442, "reach": 0.0651, "jac": -0.0055}),
}


def local_metric_anchors(tag: str, ctx: str = "Q_HepG2") -> dict[str, float]:
    """本地 per-metric anchor 归一化分数（scored.csv 的 from_replicate 列）。"""
    p = OUT / tag / f"{ctx}_scored.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    s = pd.read_csv(p).set_index("metric")
    short = {v: k for k, v in METRICS.items()}
    out = {}
    for long_name, short_name in METRICS.items():
        if long_name in s.index:
            v = s.loc[long_name, "from_replicate"]
            if pd.notna(v):
                out[short_name] = float(v)
    return out


def raw_metrics(tag: str, ctx: str = "Q_HepG2") -> dict[str, float]:
    """本地 RAW per-metric（agg mean，无 anchor 归一化）——参考值。"""
    p = OUT / tag / f"{ctx}_eval" / "agg_results.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    agg = pd.read_csv(p).set_index("statistic")
    out = {}
    for long, short in METRICS.items():
        if long in agg.columns and "mean" in agg.index:
            v = agg.loc["mean", long]
            if pd.notna(v):
                out[short] = float(v)
    return out


def local_anchor(tag: str, ctx: str = "Q_HepG2") -> float | None:
    p = OUT / tag / f"{ctx}_scored.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p)
    if "avg_score" in set(s.metric):
        return float(s.loc[s.metric == "avg_score", "from_replicate"].iloc[0])
    return None


def fit_calib() -> dict:
    """per-family × per-metric: official_raw = a·local_raw + b（点<2 则用全局）。"""
    rows = []
    for (fam, stem), rec in OFFICIAL_POINTS.items():
        try:
            lr = local_metric_anchors(rec["local"])
        except FileNotFoundError:
            print(f"[skip] {rec['local']} scored.csv 缺失")
            continue
        for m, ov in rec["official"].items():
            if m in lr:
                rows.append({"family": fam, "metric": m,
                             "local_raw": lr[m], "official_raw": ov})
    df = pd.DataFrame(rows)
    fit: dict = {"global": {}, "family": {}, "points": rows}
    if df.empty:
        CALIB.write_text(json.dumps(fit, indent=1))
        print("[warn] no calibration points resolved")
        return fit
    for m in METRICS.values():
        sub = df[df.metric == m]
        if len(sub) >= 3:
            a, b = np.polyfit(sub.local_raw, sub.official_raw, 1)
            fit["global"][m] = {"a": float(a), "b": float(b), "n": int(len(sub)),
                                "resid_std": float(np.std(sub.official_raw - (a * sub.local_raw + b)))}
        for fam in sub.family.unique():
            sf = sub[sub.family == fam]
            if len(sf) >= 2:
                a, b = np.polyfit(sf.local_raw, sf.official_raw, 1)
                fit["family"].setdefault(fam, {})[m] = {
                    "a": float(a), "b": float(b), "n": int(len(sf))}
    CALIB.write_text(json.dumps(fit, indent=1))
    print(f"wrote {CALIB}")
    return fit


def predict_official(tag: str, family: str, ctx: str = "Q_HepG2") -> dict:
    fit = json.loads(CALIB.read_text())
    lr = local_metric_anchors(tag, ctx)
    out = {}
    for m, v in lr.items():
        f = fit.get("family", {}).get(family, {}).get(m) or fit["global"].get(m)
        out[m] = round(f["a"] * v + f["b"], 4) if f else None
    known = [x for x in out.values() if x is not None]
    # 官方公式 Overall=(pds+mse+nmae+fid+reach+jac)/6, mse 官方恒 0 且无校准点 → 按 6 分母求和。
    # 历史: 此处曾用 mean(known)=sum/5, 恒定膨胀 Overall/5≈+0.017 (λ65 反演 0.5316/6=0.0886 与官方吻合)。
    out["overall_pred"] = round(float(np.sum(known)) / 6, 4) if len(known) >= 5 else None
    out["local_raw"] = {k: round(v, 4) for k, v in lr.items()}
    out["local_anchor"] = local_anchor(tag, ctx)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["raw", "calib", "predict"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--family", default="heur")
    ap.add_argument("--ctx", default="Q_HepG2")
    args = ap.parse_args()
    if args.mode == "raw":
        print(json.dumps(raw_metrics(args.tag, args.ctx), indent=1))
    elif args.mode == "calib":
        fit = fit_calib()
        for m, f in fit["global"].items():
            print(f"{m:6s}: official = {f['a']:.2f}*local + {f['b']:+.3f}  (n={f['n']}, resid sd={f['resid_std']:.3f})")
    else:
        print(json.dumps(predict_official(args.tag, args.family, args.ctx), indent=1))


if __name__ == "__main__":
    main()
