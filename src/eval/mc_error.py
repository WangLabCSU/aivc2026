"""MC 误差棒：量化 local-bench 单次 400-cell 参考子采样的分数噪声。

问题：bench 每个上下文只抽一次参考（每扰动 400 细胞 + NTC 2000，seed=0），
cell-eval2 的 DE/锚点指标全部建立在这一个子采样上——候选间 0.01-0.03 的
overall 差距可能只是参考侧采样噪声。

方法：固定 seed-0 已选的 100 个扰动（data/interim/local_bench/<ctx>/targets.json，
保证与候选 pred 的扰动集合一致、可复用 pred），只用不同 seed 重采"细胞"，
构造 K 个新参考；对同一 pred 分别跑 cell-eval2（命令行与 local_bench.run_celleval
完全一致）；加上已有 seed-0 分数共 K+1 个读数 → 每指标 mean±std。

判定规则（回答"0.02 的 overall 差距是否超出 MC 噪声"）：
  - 两个候选在同一参考上评测时共享参考侧噪声，差值的 SD ≤ sqrt(2)*SD_marginal
    （不相关上界；实际正相关会更小）。故若 0.02 < sqrt(2)*SD_overall，
    则该差距不可能因参考重采样而翻转 → 判定超出自采样 MC 噪声。
  - 若 0.02 > sqrt(2)*SD_overall，需对全部候选跑多 seed 参考 + 配对比较才能下结论。

用法:
  python src/eval/mc_error.py --tag l11_lam1.5 --context Q_HepG2 --seeds 1,2,3
输出:
  outputs/metrics/local_bench/mc_errors.json + stdout 表格
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
ENV_BIN = Path("/data/home/wsx/miniconda3/envs/aivc2026/bin")
BENCH_DIR = ROOT / "data/interim/local_bench"
OUT_DIR = ROOT / "outputs/metrics/local_bench"
MC_DIR = Path("/tmp/mc_refs")
METRICS = ["pds_cosine", "expr_mse_unbiased", "de_wilcoxon_lfc_nmae",
           "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
           "de_wilcoxon_sig_jaccard"]


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_ref(ctx: str, ccfg: dict, pert_set: list[str], seed: int,
              n_cells: int, n_ntc: int) -> Path:
    """按 local_bench.prepare_reference 协议重采参考：固定扰动集，只变细胞抽样。"""
    out = MC_DIR / f"{ctx}_seed{seed}.h5ad"
    if out.exists():
        print(f"[mc] reuse {out}", flush=True)
        return out
    a = ad.read_h5ad(ROOT / ccfg["h5ad"])
    tg = a.obs[ccfg["pert_col"]].astype(str)
    rng = np.random.default_rng(seed)
    keep_idx = []
    for p in pert_set + [ccfg["control"]]:
        idx = np.where(tg.to_numpy() == p)[0]
        take = rng.choice(idx, size=min(n_cells if p != ccfg["control"] else n_ntc, len(idx)),
                          replace=False)
        keep_idx.extend(take.tolist())
    sub = a[sorted(keep_idx)].copy()
    sub.obs = sub.obs[[ccfg["pert_col"]]].rename(columns={ccfg["pert_col"]: "target_gene"})
    MC_DIR.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(out, compression="gzip")
    print(f"[mc] ref ready: {out} {sub.shape}", flush=True)
    return out


def parse_agg(outdir: Path) -> dict:
    agg = pd.read_csv(outdir / "agg_results.csv")
    row = agg[agg["statistic"] == "mean"].iloc[0]
    cnt = agg[agg["statistic"] == "count"].iloc[0]
    out = {m: float(row[m]) for m in METRICS}
    out["_nmae_n_perts"] = float(cnt["de_wilcoxon_lfc_nmae"])
    return out


def run_celleval(pred: Path, ref: Path, ccfg: dict, outdir: Path) -> dict:
    """与 local_bench.run_celleval 逐字一致的评测命令。返回六指标 + nmae 有效扰动数。"""
    if (outdir / "agg_results.csv").exists():
        print(f"[mc] reuse eval {outdir}", flush=True)
        return parse_agg(outdir)
    cmd = [str(ENV_BIN / "cell-eval2"), "run", "-ap", str(pred), "-ar", str(ref),
           "--preset", "vcc2026", "--pert-col", "target_gene", "--control", ccfg["control"],
           "--input-type", "counts", "--set", "de.backend=pdex", "-o", str(outdir)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return parse_agg(outdir)


def read_original(tag: str, ctx: str) -> dict:
    return parse_agg(OUT_DIR / tag / f"{ctx}_eval")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-config", default=str(ROOT / "configs/local_bench/contexts.yaml"))
    ap.add_argument("--tag", default="l11_lam1.5")
    ap.add_argument("--context", default="Q_HepG2")
    ap.add_argument("--seeds", default="1,2,3")
    args = ap.parse_args()

    bench = load_yaml(Path(args.bench_config))
    ccfg = bench["contexts"][args.context]
    n_cells, n_ntc = bench["n_cells_per_pert"], bench["n_ntc_ref"]
    seeds = [int(s) for s in args.seeds.split(",")]

    pert_set = json.loads((BENCH_DIR / args.context / "targets.json").read_text())
    pred = OUT_DIR / args.tag / f"{args.context}_pred.h5ad"
    if not pred.exists():
        sys.exit(f"pred not found: {pred}")

    readings = {"seed0": read_original(args.tag, args.context)}
    for s in seeds:
        ref = build_ref(args.context, ccfg, pert_set, s, n_cells, n_ntc)
        outdir = MC_DIR / f"{args.context}_seed{s}_eval"
        readings[f"seed{s}"] = run_celleval(pred, ref, ccfg, outdir)
        print(f"[mc] seed{s}: " + " ".join(f"{m.split('de_wilcoxon_')[-1]}={readings[f'seed{s}'][m]:.4f}"
                                           for m in METRICS), flush=True)

    # 统计：每指标 mean±std(ddof=1)，overall = 六指标无权平均（口径同 bench）
    stats = {}
    for m in METRICS + ["overall_raw", "_nmae_n_perts"]:
        if m == "overall_raw":
            vals = [float(np.nanmean([r[x] for x in METRICS])) for r in readings.values()]
        else:
            vals = [r[m] for r in readings.values()]
        vals = np.asarray(vals, dtype=float)
        stats[m] = {"values": {k: (None if math.isnan(v) else v) for k, v in zip(readings, vals)},
                    "mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals, ddof=1)),
                    "n_readings": int(vals.size)}
    stats["overall_excl_nmae"] = {
        "mean": float(np.nanmean([np.nanmean([r[x] for x in METRICS if x != "de_wilcoxon_lfc_nmae"])
                                  for r in readings.values()])),
        "std": float(np.nanstd([np.nanmean([r[x] for x in METRICS if x != "de_wilcoxon_lfc_nmae"])
                                for r in readings.values()], ddof=1)),
    }

    sd_overall = stats["overall_raw"]["std"]
    threshold = math.sqrt(2) * sd_overall
    # 同一参考上评测的两个候选共享参考侧噪声，差值 SD <= sqrt(2)*SD（不相关上界）。
    # 故 0.02 > 上界 ⇒ 差距超出参考重采样噪声的最坏情形，判定真实；
    # 0.02 <= 上界 ⇒ 不能排除 MC 噪声，需多 seed 配对比较。
    verdict = ("0.02 > sqrt(2)*SD_overall={:.4f} → 差距超出参考重采样噪声的最坏情形上界，"
               "判定超出 MC 噪声（真实差异）".format(threshold)
               if 0.02 > threshold else
               "0.02 <= sqrt(2)*SD_overall={:.4f} → 0.02 差距不能排除 MC 噪声，"
               "需对候选做多 seed 参考配对比较".format(threshold))
    # 单指标 0.02 的显著性: 用同一上界逐指标判断
    per_metric_002 = {m: (0.02 > math.sqrt(2) * stats[m]["std"]) for m in METRICS}
    result = {
        "tag": args.tag, "context": args.context, "seeds": [0] + seeds,
        "fixed_pert_set": True, "n_perts": len(pert_set), "n_cells_per_pert": n_cells,
        "n_ntc_ref": n_ntc, "pred": str(pred), "readings": readings,
        "stats": stats, "mc_verdict_0p02": verdict,
        "per_metric_0p02_exceeds_noise": per_metric_002,
        "sqrt2_sd_overall": threshold,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "mc_errors.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n=== MC error bars [{args.tag} / {args.context}] "
          f"({len(readings)} readings, fixed {len(pert_set)} perts) ===")
    hdr = f"{'metric':<45}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}"
    print(hdr); print("-" * len(hdr))
    for m in METRICS + ["overall_raw", "overall_excl_nmae"]:
        st = stats[m]
        vals = sorted(v for v in st.get("values", {}).values() if v is not None)
        rng = f"{min(vals):>9.4f}{max(vals):>9.4f}" if vals else f"{'-':>9}{'-':>9}"
        print(f"{m:<45}{st['mean']:>9.4f}{st['std']:>9.4f}{rng}")
    print(f"\n[_nmae_n_perts] contributing perts per reading: "
          f"{[int(v) for v in stats['_nmae_n_perts']['values'].values() if v is not None]}")
    print(f"\n[sqrt(2)*SD_overall] = {threshold:.4f}")
    print("[0.02 exceeds noise bound per metric] "
          + " ".join(f"{m}={ok}" for m, ok in per_metric_002.items()))
    print(f"[verdict] {verdict}")
    print(f"\nwritten: {out_json}")


if __name__ == "__main__":
    main()
