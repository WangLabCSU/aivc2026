"""逐指标 affine 校准：本地 bench 归一化分数 → 官方 A/B/C 分数。

用启发式家族 λ1.5/λ2.5/λ3.5 三个完整配对点（每点 = P_H1/Q_HepG2/R_Jurkat 三上下文
归一化均值 vs 官方 A/B/C 六指标），逐指标拟合 official_m = a_m·local_m + b_m。
fusion v1/v2 不进拟合（其 Q/R 本地分为 in-sample，见 docs v4/v5）。

数据源：
  local : outputs/metrics/local_bench/<tag>/<ctx>_scored.csv 的 from_replicate 列
          （bench_score.py 产物；mse 恒 0 = clamp，无区分力）
  official: outputs/metrics/<entry>_online.json 的 per_metric/overall

用法:
  python src/eval/affine_calibration.py
产物: outputs/metrics/local_bench/affine_calibration.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LB = ROOT / "outputs/metrics/local_bench"
OM = ROOT / "outputs/metrics"

SHORT = {"pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse",
         "de_wilcoxon_lfc_nmae": "nmae",
         "de_wilcoxon_direction_fidelity_yield_raw": "fid",
         "de_wilcoxon_direction_reach_raw": "reach",
         "de_wilcoxon_sig_jaccard": "jac"}
METRICS = list(SHORT.values())
CONTEXTS = ["P_H1", "Q_HepG2", "R_Jurkat"]

# 启发式家族配对点（拟合用）+ λ1.0（仅外推 sanity，不进拟合）
HEUR_POINTS = {
    "lam1.5": {"local_tag": "l11_lam1.5",
               "official_json": "20260930_l11-sampled_lam15_online.json"},
    "lam2.5": {"local_tag": "l11_lam2.5",
               "official_json": "20261001_l16-sampled_lam25_online.json"},
    "lam3.5": {"local_tag": "l11_lam3.5",
               "official_json": "20261002_l17-sampled_lam35_online.json"},
}
EXTRA_POINTS = {
    "lam1.0": {"local_tag": "l11_lam1.0", "official_json": None},
    # fusion 家族：官方点在，但本地 3 上下文归一化含 in-sample（Q/R 训练池）→ 不进拟合
    "fusion_v1@1.5": {"local_tag": None,
                      "official_json": "20261001_l15fusion_v1_online.json"},
    "fusion_v2@2.5": {"local_tag": None,
                      "official_json": "20261002_l15fusion_v2_online.json"},
}


def local_norm(tag: str) -> dict:
    """<tag> 三上下文归一化均值（per_metric + overall，官方聚合口径）。"""
    per_metric_acc = {m: [] for m in METRICS}
    ctx_overalls = []
    for ctx in CONTEXTS:
        f = LB / tag / f"{ctx}_scored.csv"
        s = pd.read_csv(f).set_index("metric")["from_replicate"]
        for full, short in SHORT.items():
            v = s.get(full)
            if v is not None and np.isfinite(float(v)):
                per_metric_acc[short].append(float(v))
        ctx_overalls.append(float(s["avg_score"]))
    pm = {m: float(np.mean(v)) for m, v in per_metric_acc.items() if v}
    overall = float(np.mean(ctx_overalls))
    assert abs(overall - float(np.mean(list(pm.values())))) < 1e-6, \
        f"{tag}: overall 与六指标均值不一致（有指标缺锚？）"
    return {"per_metric": pm, "overall": overall,
            "per_context_overall": dict(zip(CONTEXTS, ctx_overalls))}


def official_norm(js: str) -> dict:
    d = json.load(open(OM / js))
    pm = {k: float(v) for k, v in d["per_metric"].items()}
    return {"per_metric": pm, "overall": float(d["overall"])}


def fit_affine(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """一维 affine 拟合，返回 (a, b, R²)。"""
    a, b = np.polyfit(xs, ys, 1)
    pred = a * xs + b
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")
    return float(a), float(b), r2


def loo_errors(xs: np.ndarray, ys: np.ndarray) -> list[dict]:
    """三点 leave-one-out：两点定线（无自由度），预测第三点。"""
    out = []
    for i in range(len(xs)):
        mask = np.ones(len(xs), bool)
        mask[i] = False
        a, b = np.polyfit(xs[mask], ys[mask], 1)
        pred = a * xs[i] + b
        out.append({"held_out": i, "pred": round(pred, 4),
                    "actual": round(float(ys[i]), 4),
                    "err": round(pred - float(ys[i]), 4)})
    return out


def main() -> None:
    pairs, fit_x, fit_y = {}, {}, {}
    for name, spec in HEUR_POINTS.items():
        loc = local_norm(spec["local_tag"])
        off = official_norm(spec["official_json"])
        pairs[name] = {"local": loc, "official": off}
        for m in METRICS:
            fit_x.setdefault(m, []).append(loc["per_metric"][m])
            fit_y.setdefault(m, []).append(off["per_metric"][m])
        fit_x.setdefault("overall", []).append(loc["overall"])
        fit_y.setdefault("overall", []).append(off["overall"])

    fits = {}
    for m in METRICS + ["overall"]:
        xs, ys = np.array(fit_x[m]), np.array(fit_y[m])
        if np.allclose(xs, xs[0]) and np.allclose(ys, ys[0]):  # mse: 恒 0，退化
            fits[m] = {"a": 0.0, "b": 0.0, "r2": None,
                       "loo": [{"held_out": i, "pred": 0.0, "actual": 0.0,
                                "err": 0.0} for i in range(3)],
                       "local_range": [float(xs.min()), float(xs.max())],
                       "note": "退化：本地/官方均恒 0（clamp），仿射无意义，预测恒 0"}
            continue
        a, b, r2 = fit_affine(xs, ys)
        fits[m] = {"a": round(a, 4), "b": round(b, 4),
                   "r2": round(r2, 4) if np.isfinite(r2) else None,
                   "loo": loo_errors(xs, ys),
                   "local_range": [round(float(xs.min()), 4),
                                   round(float(xs.max()), 4)]}

    # λ1.0 外推 sanity（无官方点）
    lam10 = local_norm(EXTRA_POINTS["lam1.0"]["local_tag"])
    ov = fits["overall"]
    lam10_pred = ov["a"] * lam10["overall"] + ov["b"]

    result = {
        "scale": "official anchor-normalized local bench (0=baseline, 1=replicate)",
        "method": "per-metric affine official_m = a_m*local_m + b_m; "
                  "3 heuristic points (lam1.5/2.5/3.5, each = mean of P_H1/Q_HepG2/R_Jurkat)",
        "fit_points": pairs,
        "excluded": {
            "fusion_v1/v2": "官方点存在，但本地 Q/R 归一化分为 in-sample（训练池含 "
                            "hepg2/jurkat）→ 不可作干净配对；official 见 fit_points 旁注",
            "fusion_official": {k: official_norm(v["official_json"])
                                for k, v in EXTRA_POINTS.items()
                                if v["official_json"] and "fusion" in k},
        },
        "fits": fits,
        "extrapolation_lam1.0": {
            "local_overall_3ctx": round(lam10["overall"], 4),
            "predicted_official_overall": round(lam10_pred, 4),
            "note": "外推到拟合区间（local 0.20-0.31）之下，仅参考"},
        "hct116_fid_diff_prediction": {
            "local_fid_diff_S_HCT116_fulldepth": 0.106,
            "predicted_official_fid_diff": round(fits["fid"]["a"] * 0.106, 4),
            "source": "fusion v2@1.5 fid 0.777 vs heuristic lam1.5 0.672 "
                      "(outputs/metrics/local_bench/hct116_fulldepth_norm.json)",
            "caveat": "斜率由 P/Q/R 三上下文均值拟合，S_HCT116 深度 ref 是另一上下文，"
                      "假设 slope 可迁移"},
        "usage_rules": {
            "usable_when": [
                "启发式家族（λ·Δ 结构，Δ 为跨数据集聚合签名）",
                "本地分为 P_H1/Q_HepG2/R_Jurkat 三上下文 bundle 归一化均值（同 bundle、"
                "device=cuda 与建 bundle 一致、seed-0 ref）",
                "local overall 落在拟合区间 [0.20, 0.27] 内（区间外为外推）",
            ],
            "overall": "official_overall ≈ 2.096*local - 0.526，±0.08（LOO 0.039 + ref 噪声 0.028*2.1 放大）",
            "usable_metrics": {"pds": "±0.05", "reach": "±0.01",
                               "fid": "区间放宽 ±0.10（LOO 0.194，恒定偏移规则在 fid 上失效）"},
            "unusable": {
                "nmae": "官方端平坦（0.017-0.034），R²=0.21，ref 重建噪声 ±0.233 淹没信号；"
                        "直接用官方历史值域做区间",
                "jac": "官方端平坦（-0.019~-0.039），slope 反号且无意义；同上",
                "mse": "两端 clamp 恒 0，无区分力",
                "fusion/学习式家族": "无干净配对点（本地 Q/R in-sample）；官方预期用 v4 规则 "
                                    "official ≈ local - 0.42；fid 修复换算见 hct116_fid_diff_prediction",
            },
        },
    }
    out = LB / "affine_calibration.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({m: {k: v[m][k] for k in ("a", "b", "r2")} if m in v else
                          {kk: v[m][kk] for kk in ("a", "b", "r2", "loo")}
                      for v in [fits] for m in v}, indent=2))
    print(f"\nwritten: {out}")
    print(f"lam1.0 extrapolation: local {lam10['overall']:.4f} -> official {lam10_pred:.4f}")
    print(f"hct116 fid diff +0.106 -> official {fits['fid']['a'] * 0.106:+.4f}")


if __name__ == "__main__":
    main()
