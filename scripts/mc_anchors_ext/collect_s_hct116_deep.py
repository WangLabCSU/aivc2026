"""S_HCT116 深度多重复锚点汇总：4 ref（seed 0/11/12/13）归一化分数 mean±std → mc_anchors.json。

节点 key = "S_HCT116_deep"（保留浅版 "S_HCT116" 节点）。含:
- 每 seed from_replicate 五指标 + overall5（scored_s{seed}.csv）;
  深度 ref 上 expr_mse_unbiased_capped_norm 的 replicate 锚点 > baseline（4.13 vs 2.23，
  cap/unbiased 修正预算机制所致），官方拒绝建 0..1 标度 → 深度节点降为 5 指标口径
  （同 bench_score.manual_anchor 先例）; u 腿 raw mse 仍单独存档;
- per_metric mean±std(ddof=1)、u/b/r 三腿原始值;
- 深度 vs 浅版 vs S_RPE1 的 overall ref 方差对比——三者统一重算成同 5 指标口径
  （浅版/RPE1 节点存了每 seed 每指标 normalized 值，直接重平均）;
- refs_deep_meta.json 的 ref 统计与靶集重叠。

用法: python scripts/mc_anchors_ext/collect_s_hct116_deep.py [--commit]
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
LB = ROOT / "outputs/metrics/local_bench"
DEEP_DIR = ROOT / "data/interim/local_bench/S_HCT116"
ANCHORS = LB / "mc_anchors.json"
SEEDS = [0, 11, 12, 13]
METRICS5 = ["pds_cosine", "de_wilcoxon_lfc_nmae",
            "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
            "de_wilcoxon_sig_jaccard"]
MSE = "expr_mse_unbiased_capped_norm"
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


def stat(vals: list[float]) -> dict:
    v = np.asarray(vals, dtype=float)
    return {"values": [round(float(x), 6) for x in v],
            "mean": round(float(v.mean()), 6), "std": round(float(v.std(ddof=1)), 6),
            "range": [round(float(v.min()), 6), round(float(v.max()), 6)]}


def overall5_from_node(node: dict) -> list[float]:
    """把旧节点每 seed 的 normalized 六指标重算成 5 指标（去 mse）无权平均。"""
    per_seed = node["normalized_scores_per_ref"]
    out = []
    for s in SEEDS:
        vals = per_seed[f"seed{s}"]
        out.append(float(np.mean([vals[SHORT[m]] for m in METRICS5])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    meta = json.loads((DEEP_DIR / "refs_deep_meta.json").read_text())
    readings, comps, u_mse = {}, {}, {}
    for s in SEEDS:
        bundle = LB / f"mc_anchor_bundles/S_HCT116_deep_seed{s}"
        scored = DEEP_DIR / f"scored_s{s}.csv"
        eval_dir = DEEP_DIR / f"eval_s{s}"
        for p in (bundle / "manifest.json", scored, eval_dir / "agg_results.csv"):
            if not p.exists():
                raise SystemExit(f"missing artifact for seed{s}: {p}")
        b, r = bundle_ends(bundle)
        agg = pd.read_csv(eval_dir / "agg_results.csv")
        row = agg[agg["statistic"] == "mean"].iloc[0]
        u = {m: float(row[m]) for m in METRICS5}
        # 5 指标 profile 下 run 的 agg 无 mse 列 → raw mse 无法存档（标度本就被官方拒绝）
        u_mse[f"seed{s}"] = (float(row[MSE]) if MSE in row.index and not pd.isna(row[MSE])
                             else None)
        readings[f"seed{s}"] = read_scored(scored)
        comps[f"seed{s}"] = {"u": u, "b": b, "r": r}
        print(f"seed{s}: overall5={readings[f'seed{s}']['overall5']:.4f} "
              f"(raw mse u={u_mse[f'seed{s}']})", flush=True)

    per_metric = {SHORT[m]: stat([readings[t][SHORT[m]] for t in readings]) for m in METRICS5}
    per_metric["overall5"] = stat([readings[t]["overall5"] for t in readings])

    old = json.loads(ANCHORS.read_text())
    shallow5 = stat(overall5_from_node(old["S_HCT116"]))
    rpe15 = stat(overall5_from_node(old["S_RPE1"]))
    deep_std = per_metric["overall5"]["std"]
    comparisons = {
        "scale": "overall = 5-metric (mse dropped) anchor-normalized mean, "
                 "0=generic baseline, 1=replicate anchor; 同口径三方对比",
        "shallow_S_HCT116_overall5": shallow5,
        "deep_S_HCT116_overall5": per_metric["overall5"],
        "S_RPE1_overall5": rpe15,
        "deep_over_shallow_std_ratio": round(deep_std / shallow5["std"], 3),
        "deep_vs_rpe1_std_ratio": round(deep_std / rpe15["std"], 3),
        "deep_converges_to_rpe1": bool(deep_std <= rpe15["std"] * 1.5),
    }
    node = {
        "experiment": "S_HCT116 深度多重复锚点（pooled ≥400 cells/靶 gate, 每 seed 随机 100 靶, "
                      "pred 随 ref 重建; 深度版 ref 方差检验）",
        "date": date.today().isoformat(),
        "scale": "5-metric anchor-normalized（mse 标度在深度 ref 上退化被官方拒绝, 见 notes）",
        "design": meta.get("design", {}),
        "ref_stats": {k: meta[k] for k in meta if k.startswith("seed")},
        "pred": "outputs/metrics/local_bench/l11_lam1.5/S_HCT116_pred_s{0,11,12,13}.h5ad "
                "(l11 λ1.5, ref-NTC bootstrap ⊕ 2^1.5Δ, var 轴=ref=pooled 38,606)",
        "normalized_scores_per_ref": readings,
        "per_metric_normalized": per_metric,
        "components": {"u_user_raw": {SHORT[m]: stat([c["u"][m] for c in comps.values()])
                                      for m in METRICS5},
                       "u_raw_mse_unscaled": u_mse,
                       "b_baseline_arm": {SHORT[m]: stat([c["b"][m] for c in comps.values()])
                                          for m in METRICS5},
                       "r_replicate_anchor": {SHORT[m]: stat([c["r"][m] for c in comps.values()])
                                              for m in METRICS5},
                       "raw_values": comps},
        "comparisons": comparisons,
        "notes": [
            "mse 深度退化根因: expr_mse_unbiased_capped_norm 的 replicate 端(半分 200v200)"
            "经 cap/unbiased 修正预算后仍 ~2x baseline 端(4.13 vs 2.23), r-b<0 无headroom, "
            "vcc2026 成员不给 --allow-degenerate-baseline 豁免; 处置=5 指标标度"
            "（bench_score.manual_anchor 同类先例）, u 腿 raw mse 已存档供横向相对比较",
            "浅版节点(S_HCT116)协议=固定 seed-0 靶集只变细胞抽样; 深度版因浅版靶集在源中多"
            "<400 细胞无法沿用, 改为每 seed 独立抽 100 靶(总方差含靶集抽样成分), "
            "4 seed 靶集两两重叠仅 6-13/100",
            "评分口径与官方一致: 0=generic baseline, 1=replicate anchor(5 disjoint half-split 均值)",
        ],
    }
    old["S_HCT116_deep"] = node
    tmp = ANCHORS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(old, indent=1, ensure_ascii=False))
    tmp.replace(ANCHORS)
    print(json.dumps({"deep_overall5": per_metric["overall5"], "shallow5": shallow5,
                      "rpe1_5": rpe15, "comparisons": comparisons}, indent=2))
    if args.commit:
        import subprocess
        subprocess.run(["git", "add", "src/eval/rebuild_s_hct116_refs.py",
                        "scripts/mc_anchors_ext/pipeline_s_hct116_deep.sh",
                        "scripts/mc_anchors_ext/collect_s_hct116_deep.py"], cwd=ROOT, check=True)
        rc = subprocess.run(["git", "commit", "-m",
                             f"S_HCT116 深度多重复锚点: 4 ref(≥400 gate) 归一化 overall5 "
                             f"{per_metric['overall5']['mean']:.4f}±{deep_std:.4f} vs 浅版 "
                             f"±{shallow5['std']:.4f}/RPE1 ±{rpe15['std']:.4f}; "
                             f"mse 标度深度退化→5指标口径"], cwd=ROOT)
        if rc.returncode:
            print("git commit skipped (nothing to commit or failed)")
    print("written:", ANCHORS)


if __name__ == "__main__":
    main()
