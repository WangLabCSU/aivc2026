"""S_HCT116 pooled 固定靶集+固定 pred 纯 ref 重抽汇总: 4 ref（seed 0/11/12/13, 靶集/targets.json,
pred 固定共享）归一化分数 mean±std → mc_anchors.json 节点 "S_HCT116_pooled_fixed"。

归因链终环: S_HCT116_deep ±0.075（靶集重抽+pred 重建+ref 重抽）→ S_HCT116_fixed ±0.092
（靶集固定, pred 仍随 seed 重建）→ 本节点 std = 纯 ref 细胞重抽（含 eval baseline/anchor 腿
随 ref 重算）方差。判读规则模板: std<=0.03 → 残差为锚点标度噪声, 判读线 <0.08 修订为 <0.03;
std>=0.05 → 结构性, 规则维持。

用法: python scripts/mc_anchors_ext/collect_s_hct116_pooled_fixed.py [--commit]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
LB = ROOT / "outputs/metrics/local_bench"
PFIX_DIR = ROOT / "data/interim/local_bench/S_HCT116"
ANCHORS = LB / "mc_anchors.json"
SEEDS = [0, 11, 12, 13]
METRICS5 = ["pds_cosine", "de_wilcoxon_lfc_nmae",
            "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
            "de_wilcoxon_sig_jaccard"]
SHORT = {"pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse",
         "de_wilcoxon_lfc_nmae": "nmae", "de_wilcoxon_direction_fidelity_yield_raw": "fid",
         "de_wilcoxon_direction_reach_raw": "reach", "de_wilcoxon_sig_jaccard": "jac"}


def bundle_ends(bundle: Path) -> tuple[dict, dict]:
    b = pd.read_csv(bundle / "baseline_agg.csv")
    brow = b[b["statistic"] == "mean"].iloc[0]
    r = pd.read_parquet(bundle / "anchor_agg.parquet").set_index("metric")["replicate"]
    return ({m: float(brow[m]) for m in METRICS5},
            {m: float(r[m]) for m in METRICS5})


def read_scored(path: Path) -> dict:
    s = pd.read_csv(path).set_index("metric")
    out = {SHORT[m]: float(s.loc[m, "from_replicate"]) for m in METRICS5}
    out["overall5"] = float(s.loc["avg_score", "from_replicate"])
    return out


def stat(vals) -> dict:
    v = np.asarray(vals, dtype=float)
    return {"values": [round(float(x), 6) for x in v],
            "mean": round(float(v.mean()), 6), "std": round(float(v.std(ddof=1)), 6),
            "range": [round(float(v.min()), 6), round(float(v.max()), 6)]}


def overall5_from_node(node: dict) -> list[float]:
    per_seed = node["normalized_scores_per_ref"]
    return [float(np.mean([per_seed[f"seed{s}"][SHORT[m]] for m in METRICS5])) for s in SEEDS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    meta = json.loads((PFIX_DIR / "refs_pooled_fixed_meta.json").read_text())
    readings, comps = {}, {}
    for s in SEEDS:
        bundle = LB / f"mc_anchor_bundles/S_HCT116_pooled_fixed_seed{s}"
        scored = PFIX_DIR / f"scored_pfix_s{s}.csv"
        agg_path = PFIX_DIR / f"eval_pfix_s{s}/agg_results.csv"
        for p in (bundle / "manifest.json", scored, agg_path):
            if not p.exists():
                raise SystemExit(f"missing artifact for seed{s}: {p}")
        b, r = bundle_ends(bundle)
        agg = pd.read_csv(agg_path)
        row = agg[agg["statistic"] == "mean"].iloc[0]
        u = {m: float(row[m]) for m in METRICS5}
        readings[f"seed{s}"] = read_scored(scored)
        comps[f"seed{s}"] = {"u": u, "b": b, "r": r}
        print(f"seed{s}: overall5={readings[f'seed{s}']['overall5']:.4f}", flush=True)

    per_metric = {SHORT[m]: stat([readings[t][SHORT[m]] for t in readings]) for m in METRICS5}
    per_metric["overall5"] = stat([readings[t]["overall5"] for t in readings])

    old = json.loads(ANCHORS.read_text())
    deep5 = stat(overall5_from_node(old["S_HCT116_deep"]))
    fix5 = stat(overall5_from_node(old["S_HCT116_fixed"]))
    shallow5 = stat(overall5_from_node(old["S_HCT116"]))
    rpe15 = stat(overall5_from_node(old["S_RPE1"]))
    pfix_std = per_metric["overall5"]["std"]
    if pfix_std <= 0.03:
        verdict = ("纯 ref 重抽 std<=0.03 → 残差为锚点标度噪声; S_HCT116 判读线 <0.08 "
                   "修订为 <0.03（同 ref 协议重跑即可复现读数）")
    elif pfix_std >= 0.05:
        verdict = ("纯 ref 重抽 std>=0.05 → ref 侧方差结构性（非靶集/非 pred 重建所致）; "
                   "S_HCT116 判读线 <0.08 维持")
    else:
        verdict = ("0.03<std<0.05 中间带 → 不可判, 需 n>4 或配对设计复核")
    comparisons = {
        "scale": "overall = 5-metric (mse dropped) anchor-normalized mean, "
                 "0=generic baseline, 1=replicate anchor; 同口径五方对比",
        "pooled_fixed_overall5": per_metric["overall5"],
        "fixed_targets0_predrebuild_overall5": fix5,
        "deep_overall5": deep5,
        "shallow_overall5": shallow5,
        "S_RPE1_overall5": rpe15,
        "std_ratio_pfix_vs_fixed": round(pfix_std / fix5["std"], 3),
        "std_ratio_pfix_vs_deep": round(pfix_std / deep5["std"], 3),
        "attribution": ("var(S_HCT116_pooled_fixed)=纯 ref 细胞重抽(+eval 腿重算); "
                        "var(fixed)-var(pooled_fixed) 差额=pred NTC-bootstrap 重建+靶集差异"
                        "(targets_s0 vs targets.json)成分" if pfix_std < fix5["std"] else
                        "纯 ref 重抽已复现 fixed/deep 量级 → ref 侧即方差主体"),
        "verdict": verdict,
    }
    fix_pm = old["S_HCT116_fixed"]["per_metric_normalized"]
    per_metric_cmp = {m: {"fixed_std": fix_pm[m]["std"], "pooled_fixed_std": per_metric[m]["std"]}
                      for m in per_metric}
    node = {
        "experiment": "S_HCT116 pooled 固定靶集+固定 pred 纯 ref 重抽（归因确认终版: "
                      "剥去靶集重抽与 pred 重建后, 剩余 ref 方差量级）",
        "date": date.today().isoformat(),
        "scale": "5-metric anchor-normalized（同 deep/fixed 节点, mse 标度深度退化不用）",
        "design": meta.get("design", {}),
        "ref_stats": {k: meta[k] for k in meta if k.startswith("seed")},
        "pred": "outputs/metrics/local_bench/l11_lam1.5/S_HCT116_pred.h5ad 固定共享"
                "（40,400×38,606, 轴/扰动集与 pooled 全量版及 targets.json 校验一致）",
        "normalized_scores_per_ref": readings,
        "per_metric_normalized": per_metric,
        "per_metric_std_fixed_vs_pooled_fixed": per_metric_cmp,
        "components": {"u_user_raw": {SHORT[m]: stat([c["u"][m] for c in comps.values()])
                                      for m in METRICS5},
                       "b_baseline_arm": {SHORT[m]: stat([c["b"][m] for c in comps.values()])
                                          for m in METRICS5},
                       "r_replicate_anchor": {SHORT[m]: stat([c["r"][m] for c in comps.values()])
                                              for m in METRICS5},
                       "raw_values": comps},
        "comparisons": comparisons,
        "notes": [
            "源勘误: HCT116_pooled.h5ad 现文件即 109 批全量版（3,409,169×38,606, "
            ".pooled_full.done n_batches=109）; 任务提示 '796,139 cells=27 批版' 为旧部分构建。",
            "设计偏离: 任务原文 '每靶 30 细胞'→改为 request-400-cap-avail（~19.2k cells/ref, "
            "与任务背景 '~19.2k cells' 及浅版 bench 协议一致; 30/靶仅 ~5k cells, 与任务自身"
            " '~30-50k'/'~42k' 规模提示矛盾）。gate>=30: 100/100 过门（min=30, median=139.5）。",
            "pred 不随 seed 重建: 轴（38,606 同序）与扰动集（NTC+100 靶）已断言校验 → "
            "4 读数差异只能来自 ref 细胞重抽及其触发的 eval baseline/anchor 腿重算。",
        ],
    }
    old["S_HCT116_pooled_fixed"] = node
    tmp = ANCHORS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(old, indent=1, ensure_ascii=False))
    tmp.replace(ANCHORS)
    print(json.dumps({"pooled_fixed_overall5": per_metric["overall5"], "fixed5": fix5,
                      "deep5": deep5, "shallow5": shallow5, "rpe1_5": rpe15,
                      "comparisons": comparisons}, indent=2, ensure_ascii=False))
    if args.commit:
        import subprocess
        subprocess.run(["git", "add",
                        "scripts/mc_anchors_ext/build_refs_s_hct116_pooled_fixed.py",
                        "scripts/mc_anchors_ext/pipeline_s_hct116_pooled_fixed.sh",
                        "scripts/mc_anchors_ext/collect_s_hct116_pooled_fixed.py"],
                       cwd=ROOT, check=True)
        rc = subprocess.run(["git", "commit", "-m",
                             f"S_HCT116 pooled 固定靶集+固定 pred 纯 ref 重抽: overall5 "
                             f"{per_metric['overall5']['mean']:.4f}±{pfix_std:.4f} vs fixed "
                             f"±{fix5['std']:.4f}/deep ±{deep5['std']:.4f} → {verdict}"],
                            cwd=ROOT)
        if rc.returncode:
            print("git commit skipped (nothing to commit or failed)")
    print("written:", ANCHORS)


if __name__ == "__main__":
    main()
