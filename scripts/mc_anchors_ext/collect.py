"""E3 汇总：把 P_H1/R_Jurkat 多重复锚点结果合并进 mc_anchors.json（保留 Q_HepG2 节点）。

数据源（每 ctx）：seed0 = l11_lam1.5/<ctx>_scored.csv + bundles/<ctx> + l11_lam1.5/<ctx>_eval；
seed{11,12,13} = /tmp/mc_anchors_ext/*_scored.csv + mc_anchor_bundles/<ctx>_seed{S}。
含 u-only 传播方差分解（公式自算，seed-0 官方读数自检）。
用法: python collect.py --contexts P_H1,R_Jurkat
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
LB = ROOT / "outputs/metrics/local_bench"
TMP = Path("/tmp/mc_anchors_ext")
SEEDS = [0, 11, 12, 13]
METRICS = ["pds_cosine", "expr_mse_unbiased_capped_norm", "de_wilcoxon_lfc_nmae",
           "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
           "de_wilcoxon_sig_jaccard"]
SHORT = {"pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse",
         "de_wilcoxon_lfc_nmae": "nmae", "de_wilcoxon_direction_fidelity_yield_raw": "fid",
         "de_wilcoxon_direction_reach_raw": "reach", "de_wilcoxon_sig_jaccard": "jac"}
CONTROL = {"P_H1": "non-targeting", "R_Jurkat": "non-targeting", "Q_HepG2": "non-targeting"}


def bundle_ends(bundle: Path) -> tuple[dict, dict]:
    b = pd.read_csv(bundle / "baseline_agg.csv")
    brow = b[b["statistic"] == "mean"].iloc[0]
    r = pd.read_parquet(bundle / "anchor_agg.parquet").set_index("metric")["replicate"]
    return ({m: float(brow[m]) for m in METRICS}, {m: float(r[m]) for m in METRICS})


def read_scored(path: Path) -> dict:
    s = pd.read_csv(path).set_index("metric")
    out = {SHORT[m]: float(s.loc[m, "from_replicate"]) for m in METRICS}
    out["overall"] = float(s.loc["avg_score", "from_replicate"])
    return out


def stat(vals: list[float]) -> dict:
    v = np.asarray(vals, dtype=float)
    return {"values": [round(float(x), 6) for x in v],
            "mean": round(float(v.mean()), 6), "std": round(float(v.std(ddof=1)), 6),
            "range": [round(float(v.min()), 6), round(float(v.max()), 6)]}


def collect_ctx(ctx: str) -> dict:
    readings, comps, timing = {}, {}, {}
    for s in SEEDS:
        if s == 0:
            bundle = LB / f"bundles/{ctx}"
            eval_dir = LB / f"l11_lam1.5/{ctx}_eval"
            scored = LB / f"l11_lam1.5/{ctx}_scored.csv"
            ref_path = ROOT / f"data/interim/local_bench/{ctx}/ref.h5ad"
        else:
            bundle = LB / f"mc_anchor_bundles/{ctx}_seed{s}"
            eval_dir = TMP / f"eval_{ctx}_seed{s}"
            scored = TMP / f"{ctx}_seed{s}_scored.csv"
            ref_path = TMP / f"{ctx}_seed{s}.h5ad"
        if not (bundle / "manifest.json").exists() or not scored.exists() \
                or not (eval_dir / "agg_results.csv").exists():
            print(f"[collect] {ctx} seed{s}: artifacts incomplete, skipped", flush=True)
            continue
        b, r = bundle_ends(bundle)
        agg = pd.read_csv(eval_dir / "agg_results.csv")
        u = {m: float(agg[agg["statistic"] == "mean"].iloc[0][m]) for m in METRICS}
        readings[f"seed{s}"] = read_scored(scored)
        comps[f"seed{s}"] = {"u": u, "b": b, "r": r}
    if len(readings) < 3:
        raise RuntimeError(f"{ctx}: only {len(readings)} complete readings, need >=3")
        import anndata as ad
        a = ad.read_h5ad(ref_path, backed="r")
        vc = a.obs["target_gene"].astype(str).value_counts()
        pp = vc[vc.index != CONTROL[ctx]]
        timing[f"seed{s}"] = {"n_cells": int(a.n_obs), "median_cells_per_pert": float(pp.median()),
                              "min_cells_per_pert": int(pp.min()), "n_perts_at_400": int((pp >= 400).sum())}
        a.file.close()
    per_metric_norm = {SHORT[m]: stat([readings[t][SHORT[m]] for t in readings]) for m in METRICS}
    per_metric_norm["overall"] = stat([readings[t]["overall"] for t in readings])
    per_metric_u = {SHORT[m]: stat([comps[t]["u"][m] for t in comps]) for m in METRICS}
    per_metric_b = {SHORT[m]: stat([comps[t]["b"][m] for t in comps]) for m in METRICS}
    per_metric_r = {SHORT[m]: stat([comps[t]["r"][m] for t in comps]) for m in METRICS}
    # u-only 传播（线性公式，seed-0 自检）
    chk = read_scored(LB / f"l11_lam1.5/{ctx}_scored.csv")
    f0 = {}
    for m in METRICS:
        v = (comps["seed0"]["u"][m] - comps["seed0"]["b"][m]) / \
            (comps["seed0"]["r"][m] - comps["seed0"]["b"][m])
        # 官方 scoring policy: expr_mse_unbiased_capped_norm clamp_low=0（P_H1 上 mse 线性值为负，
        # scored=0.0；Q 上恒正故未暴露）。avg_score 用 clamp 后的六指标均值（对 P_H1 seed-0 复算验证）。
        f0[SHORT[m]] = max(0.0, v) if m == "expr_mse_unbiased_capped_norm" else v
    f0["overall"] = sum(f0.values()) / 6
    assert all(abs(f0[k] - chk[k]) < 1e-6 for k in list(SHORT.values()) + ["overall"]), \
        f"{ctx}: linear formula mismatch on seed-0"
    u_only = {"seed0": f0}
    b0, r0 = comps["seed0"]["b"], comps["seed0"]["r"]
    for t in [s for s in SEEDS[1:] if f"seed{s}" in comps]:
        u_i = comps[f"seed{t}"]["u"]
        vals = {}
        for m in METRICS:
            v = (u_i[m] - b0[m]) / (r0[m] - b0[m])
            vals[SHORT[m]] = max(0.0, v) if m == "expr_mse_unbiased_capped_norm" else v
        vals["overall"] = sum(vals.values()) / 6
        u_only[f"seed{t}"] = vals
    u_only_stats = {SHORT[m]: stat([u_only[t][SHORT[m]] for t in u_only]) for m in METRICS}
    u_only_stats["overall"] = stat([u_only[t]["overall"] for t in u_only])
    decomp = {}
    for k in list(SHORT.values()) + ["overall"]:
        var_tot = per_metric_norm[k]["std"] ** 2
        var_u = u_only_stats[k]["std"] ** 2
        decomp[k] = {"std_total": per_metric_norm[k]["std"],
                     "std_u_propagated": u_only_stats[k]["std"],
                     "std_anchor_side_residual": round(float(np.sqrt(max(var_tot - var_u, 0.0))), 6),
                     "u_share_of_variance": round(var_u / var_tot, 3) if var_tot > 0 else None}
    return {
        "experiment": f"multi-replicate anchors: {ctx} bundle rebuilt from 3 re-seeded refs + seed-0",
        "date": time.strftime("%Y-%m-%d"),
        "pred": str(LB / f"l11_lam1.5/{ctx}_pred.h5ad"),
        "design": {"n_refs": len(readings), "seeds": [int(k[4:]) for k in readings],
                   "pert_set": "fixed = seed-0 targets.json (100 perts; cell-eval2 requires pred/real pert-set identity)",
                   "varied": "cell + NTC subsampling seed (protocol identical to mc_error.build_ref)",
                   "n_cells_per_pert_requested": 400, "n_ntc": 2000,
                   "anchor_splits_per_bundle": 5, "anchor_base_seed": 0,
                   "ref_stats": timing},
        "normalized_scores_per_ref": readings,
        "per_metric_normalized": per_metric_norm,
        "components": {"u_user_raw": per_metric_u, "b_baseline_arm": per_metric_b,
                       "r_replicate_anchor": per_metric_r,
                       "raw_values": {k0: {k: {kk: round(vv, 6) for kk, vv in d.items()}
                              for k, d in comps[k0].items()} for k0 in comps}},
        "u_only_propagated_vs_seed0_bundle": u_only_stats,
        "variance_decomposition": decomp,
        "verdict_seed0_vs_3ref_mean": {
            "seed0_overall": readings["seed0"]["overall"],
            "mean3_overall": round(float(np.mean([v["overall"] for k, v in readings.items() if k != "seed0"])), 6),
            "diff": round(readings["seed0"]["overall"] -
                          float(np.mean([v["overall"] for k, v in readings.items() if k != "seed0"])), 6)},
    }


ap = argparse.ArgumentParser()
ap.add_argument("--contexts", default="P_H1,R_Jurkat")
args = ap.parse_args()
out_path = LB / "mc_anchors.json"
merged = json.loads(out_path.read_text()) if out_path.exists() else {}
# 旧文件为 Q_HepG2 扁平结构 → 迁移为按上下文嵌套
if "experiment" in merged:
    merged = {"Q_HepG2": merged}
summary = {}
for ctx in args.contexts.split(","):
    node = collect_ctx(ctx)
    merged[ctx] = node
    summary[ctx] = {"overall_mean": node["per_metric_normalized"]["overall"]["mean"],
                    "overall_std": node["per_metric_normalized"]["overall"]["std"],
                    "seed0_vs_3ref_diff": node["verdict_seed0_vs_3ref_mean"]["diff"]}
merged["_cross_context_verdict"] = {
    "ref_selection_overall_std_by_context": {c: summary[c]["overall_std"] for c in summary},
    "note": "E3 检验 Q_HepG2 ±0.028 的 ref 方差是否普适；std>0.02 的上下文单 ref 差距 <0.04 视为平手"}
out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2))
print(f"wrote {out_path}")
