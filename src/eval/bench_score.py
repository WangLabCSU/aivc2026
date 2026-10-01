"""anchor-normalized bench 打分：把 local-bench 的 raw 六指标映射到官方 0..1 标度。

官方口径（cell-eval2 --preset vcc2026 的 real-bundle 链路）：
  0 = 该上下文 generic-response baseline（0 端）
  1 = 真实重复实验一致性（replicate anchor，1 端，5 个 disjoint half-split 的均值）
  per-metric: from_replicate = (u - b) / (r - b)（lower-is-better 同式，经 Scoring policy）
  overall    = 六指标 from_replicate 无权平均 → 上下文无权平均

链路（每上下文一次，产物缓存在 outputs/metrics/local_bench/bundles/<ctx>）：
  1) cell-eval2 baseline -ar ref.h5ad --preset vcc2026 --pert-col target_gene \
       --control non-targeting --input-type counts --set de.backend=pdex \
       -o bundle_build/<ctx>_baseline --save-pred bundle_build/<ctx>_baseline_pred.h5ad
  2) cell-eval2 prep-real-bundle --real ref.h5ad --baseline <baseline_pred.h5ad> \
       -o bundles/<ctx> --id <ctx>_local <同上 config 旗标>
     （内部：baseline leg 重打分 + anchor leg 5 个 disjoint half-split 全指标运行）
之后每个候选（已跑过 local_bench 的 <tag>/<ctx>_eval）只需：
  3) cell-eval2 score --user-agg <tag>/<ctx>_eval/agg_results.csv \
       --real-bundle bundles/<ctx> -o <tag>/<ctx>_scored.csv
  4) 汇总 from_replicate 六指标 + avg_score → anchor_normalized.json

用法:
  python src/eval/bench_score.py --tag l11_lam1.5 [--contexts P_H1,Q_HepG2,R_Jurkat] \
      [--build-bundles] [--n-perts 100]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_BIN = Path("/data/home/wsx/miniconda3/envs/aivc2026/bin")
BENCH_DIR = ROOT / "data/interim/local_bench"
OUT_DIR = ROOT / "outputs/metrics/local_bench"
METRICS = ["pds_cosine", "expr_mse_unbiased_capped_norm", "de_wilcoxon_lfc_nmae",
           "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
           "de_wilcoxon_sig_jaccard"]
SHORT = {"pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse",
         "de_wilcoxon_lfc_nmae": "nmae",
         "de_wilcoxon_direction_fidelity_yield_raw": "fid",
         "de_wilcoxon_direction_reach_raw": "reach",
         "de_wilcoxon_sig_jaccard": "jac"}
CFG = ["--preset", "vcc2026", "--pert-col", "target_gene",
       "--input-type", "counts", "--set", "de.backend=pdex"]


def _cfg_for(ctx: str) -> list[str]:
    """每上下文 control 标签可不同（如 HCT116 为 'Non-Targeting'）。"""
    import yaml
    ccfg = yaml.safe_load(open(ROOT / "configs/local_bench/contexts.yaml"))["contexts"][ctx]
    return CFG + ["--control", ccfg["control"]]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_bundle(ctx: str, n_perts: int = 100) -> Path:
    """构建一个上下文的 real bundle（baseline 0 端 + replicate anchor 1 端），缓存。"""
    bundle = OUT_DIR / "bundles" / ctx
    if (bundle / "manifest.json").exists():
        return bundle
    ref = BENCH_DIR / ctx / "ref.h5ad"
    build = OUT_DIR / "bundle_build"
    build.mkdir(parents=True, exist_ok=True)
    _run([str(ENV_BIN / "cell-eval2"), "baseline", "-ar", str(ref), *_cfg_for(ctx),
          "-o", str(build / f"{ctx}_baseline"),
          "--save-pred", str(build / f"{ctx}_baseline_pred.h5ad")])
    _run([str(ENV_BIN / "cell-eval2"), "prep-real-bundle", "--real", str(ref),
          "--baseline", str(build / f"{ctx}_baseline_pred.h5ad"),
          "-o", str(bundle), "--id", f"{ctx}_local", *_cfg_for(ctx)])
    return bundle


def score_tag_ctx(tag: str, ctx: str, bundle: Path) -> dict:
    """对已跑过 local_bench 的 <tag>/<ctx>_eval 做 bundle 归一化打分，返回官方标度分数。"""
    eval_dir = OUT_DIR / tag / f"{ctx}_eval"
    agg = eval_dir / "agg_results.csv"
    if not agg.exists():
        raise FileNotFoundError(f"{agg} 不存在：先跑 src/eval/local_bench.py --tag {tag}")
    scored_path = OUT_DIR / tag / f"{ctx}_scored.csv"
    _run([str(ENV_BIN / "cell-eval2"), "score", "--user-agg", str(agg),
          "--real-bundle", str(bundle), "-o", str(scored_path)])

    import pandas as pd
    scored = pd.read_csv(scored_path).set_index("metric")
    # 竞赛 bundle 注册后 avg_score 行 = from_replicate 的六指标均值（from_baseline 被置空）
    per_metric = {SHORT[m]: float(scored.loc[m, "from_replicate"])
                  for m in METRICS if m in scored.index}
    overall = float(scored.loc["avg_score", "from_replicate"])
    # 自检：overall 应等于六指标无权平均
    manual = sum(per_metric.values()) / len(per_metric)
    assert abs(overall - manual) < 1e-9, f"avg_score {overall} != mean(metrics) {manual}"
    raw = pd.read_csv(agg)
    row = raw[raw["statistic"] == "mean"].iloc[0]
    per_metric_raw = {SHORT[m]: float(row[m]) for m in METRICS if m in raw.columns}
    return {"per_metric": per_metric, "per_metric_raw": per_metric_raw, "overall": overall}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", required=True, help="逗号分隔的 local_bench tag（目录名）")
    ap.add_argument("--contexts", default="P_H1,Q_HepG2,R_Jurkat")
    ap.add_argument("--build-bundles", action="store_true",
                    help="bundle 不存在时现场构建（baseline + anchor，较慢）")
    ap.add_argument("--out", default=str(OUT_DIR / "anchor_normalized.json"))
    args = ap.parse_args()

    contexts = args.contexts.split(",")
    bundles = {}
    for ctx in contexts:
        bundles[ctx] = build_bundle(ctx) if args.build_bundles else OUT_DIR / "bundles" / ctx

    result: dict = {"scale": "official anchor-normalized (0=baseline, 1=replicate)",
                    "per_tag": {}}
    for tag in args.tags.split(","):
        per_ctx = {ctx: score_tag_ctx(tag, ctx, bundles[ctx]) for ctx in contexts}
        overall = {}
        for key in ("per_metric", "per_metric_raw"):
            overall[key] = {SHORT.get(m, m): float(sum(per_ctx[c][key][SHORT.get(m, m)]
                             for c in contexts) / len(contexts))
                            for m in METRICS}
        overall["overall"] = float(sum(per_ctx[c]["overall"] for c in contexts) / len(contexts))
        result["per_tag"][tag] = {"per_context": per_ctx, "overall": overall}
        print(f"[{tag}] overall(official)={overall['overall']:.4f} "
              + " ".join(f"{k}={v:.4f}" for k, v in overall["per_metric"].items()), flush=True)

    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
