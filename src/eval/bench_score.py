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


def build_bundle(ctx: str, n_perts: int = 100, allow_degenerate: bool = False) -> Path:
    """构建一个上下文的 real bundle（baseline 0 端 + replicate anchor 1 端），缓存。

    allow_degenerate: 传 --allow-degenerate-baseline 给 baseline 步（全深度 HCT116 ref
    的 NTC 均值 baseline 可能使 nmae 锚点退化；prep-real-bundle 若仍拒收 → 调用方走
    manual anchor 5 指标口径）。
    """
    bundle = OUT_DIR / "bundles" / ctx
    if (bundle / "manifest.json").exists():
        return bundle
    ref = BENCH_DIR / ctx / "ref.h5ad"
    build = OUT_DIR / "bundle_build"
    build.mkdir(parents=True, exist_ok=True)
    base_cmd = [str(ENV_BIN / "cell-eval2"), "baseline", "-ar", str(ref), *_cfg_for(ctx),
                "-o", str(build / f"{ctx}_baseline"),
                "--save-pred", str(build / f"{ctx}_baseline_pred.h5ad")]
    if allow_degenerate:
        base_cmd.append("--allow-degenerate-baseline")
    _run(base_cmd)
    _run([str(ENV_BIN / "cell-eval2"), "prep-real-bundle", "--real", str(ref),
          "--baseline", str(build / f"{ctx}_baseline_pred.h5ad"),
          "-o", str(bundle), "--id", f"{ctx}_local", "--force", *_cfg_for(ctx)])
    return bundle


def manual_anchor(ctx: str, n_splits: int = 5) -> dict:
    """prep-real-bundle 拒收 degenerate 时的 5 指标口径回退（nmae 标注 unavailable）。

    0 端 b_m = bundle_build/<ctx>_baseline/agg_results.csv（baseline 自评 vs 全 ref）
    1 端 r_m = 每 pert 细胞随机对半 split（5 次），一半作 pred 一半作 ref 的
              cell-eval2 自洽均值（replicate-anchor 的本地代理）。
    from_replicate = (u - b) / (r - b)，u = 候选 <tag>/<ctx>_eval。
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    ref_path = BENCH_DIR / ctx / "ref.h5ad"
    # baseline 步输出名: cell-eval2 baseline → baseline_agg.csv（allow-degenerate 时
    # nmae 列为 NaN，count=0 → 自然落入 unavailable）
    cands = [OUT_DIR / "bundle_build" / f"{ctx}_baseline" / "agg_results.csv",
             OUT_DIR / "bundle_build" / f"{ctx}_baseline" / "baseline_agg.csv"]
    bpath = next((c for c in cands if c.exists()), None)
    if bpath is None:
        raise FileNotFoundError(cands[0])
    brow = pd.read_csv(bpath)
    brow = brow[brow["statistic"] == "mean"].iloc[0]
    b = {SHORT[m]: (float(brow[m]) if (m in brow.index and pd.notna(brow[m])) else np.nan)
         for m in METRICS}
    a = ad.read_h5ad(ref_path)
    X = sp.csr_matrix(a.X)
    tg = a.obs["target_gene"].astype(str).to_numpy()
    del a
    control = _cfg_for(ctx)[-1]
    acc = {SHORT[m]: [] for m in METRICS}
    for k in range(n_splits):
        rng = np.random.default_rng(100 + k)
        keep_a: list[int] = []
        keep_b: list[int] = []
        for p in pd.unique(tg):
            idx = np.where(tg == p)[0]
            perm = rng.permutation(idx)
            h = len(perm) // 2
            keep_a.extend(perm[:h].tolist())
            keep_b.extend(perm[h:2 * h].tolist())
        pa_dir = OUT_DIR / "bundle_build" / f"{ctx}_split{k}_a"
        pa_dir.mkdir(parents=True, exist_ok=True)
        pred = ad.AnnData(X=X[sorted(keep_a)].copy(), obs=pd.DataFrame(
            {"target_gene": tg[sorted(keep_a)]},
            index=[f"s{k}_{i}" for i in range(len(keep_a))]))
        pp = pa_dir / "pred.h5ad"
        # var_names 从 ref 文件头直读（a 已释放）
        import h5py
        with h5py.File(ref_path, "r") as f:
            var_names = [x.decode() if isinstance(x, bytes) else str(x)
                         for x in f["var/_index"][:]]
        pred.var_names = pd.Index(var_names)
        pred.var.index.name = None
        pred.write_h5ad(pp)
        refk = ad.AnnData(X=X[sorted(keep_b)].copy(), obs=pd.DataFrame(
            {"target_gene": tg[sorted(keep_b)]},
            index=[f"r{k}_{i}" for i in range(len(keep_b))]))
        refk.var_names = pd.Index(var_names)
        refk.var.index.name = None
        rp = pa_dir / "ref_split.h5ad"
        refk.write_h5ad(rp)
        outk = pa_dir / "eval"
        # --config anchor5: 去掉 expr_mse_unbiased_capped_norm（对半自洽时分母≤0 会崩），
        # 见 configs/celleval/anchor5.yaml 头注
        _run([str(ENV_BIN / "cell-eval2"), "run", "-ap", str(pp), "-ar", str(rp),
              "--config", str(ROOT / "configs/celleval/anchor5.yaml"),
              "--pert-col", "target_gene", "--control", control,
              "--input-type", "counts", "--set", "de.backend=pdex", "-o", str(outk)])
        row = pd.read_csv(outk / "agg_results.csv")
        row = row[row["statistic"] == "mean"].iloc[0]
        for m in METRICS:
            if m in row.index and pd.notna(row[m]):
                acc[SHORT[m]].append(float(row[m]))
        print(f"[manual-anchor {ctx}] split {k+1}/{n_splits} done", flush=True)
    r = {kk: float(np.mean(vv)) if vv else np.nan for kk, vv in acc.items()}
    return {"b": b, "r": r}


def score_tag_ctx(tag: str, ctx: str, bundle: Path | None = None,
                  manual: dict | None = None) -> dict:
    """对已跑过 local_bench 的 <tag>/<ctx>_eval 做 bundle 归一化打分，返回官方标度分数。

    bundle=None 且 manual 给定（{b, r}）→ manual anchor 5 指标口径（nmae 缺锚即标注
    unavailable）。bundle 路径下若个别指标 from_replicate 为 NaN（degenerate 锚点），
    同样跳过并标注。
    """
    eval_dir = OUT_DIR / tag / f"{ctx}_eval"
    agg = eval_dir / "agg_results.csv"
    if not agg.exists():
        raise FileNotFoundError(f"{agg} 不存在：先跑 src/eval/local_bench.py --tag {tag}")

    import numpy as np
    import pandas as pd
    raw = pd.read_csv(agg)
    row = raw[raw["statistic"] == "mean"].iloc[0]
    per_metric_raw = {SHORT[m]: float(row[m]) for m in METRICS if m in raw.columns}

    if manual is not None:
        b, r = manual["b"], manual["r"]
        per_metric, used = {}, []
        for m in METRICS:
            s = SHORT[m]
            if s in per_metric_raw and s in b and s in r \
                    and np.isfinite(b[s]) and np.isfinite(r[s]) and abs(r[s] - b[s]) > 1e-12:
                per_metric[s] = (per_metric_raw[s] - b[s]) / (r[s] - b[s])
                used.append(s)
        missing = [SHORT[m] for m in METRICS if SHORT[m] not in per_metric]
        overall = float(sum(per_metric.values()) / len(per_metric)) if per_metric else float("nan")
        return {"per_metric": per_metric, "per_metric_raw": per_metric_raw,
                "overall": overall, "bundle_mode": "manual_halfsplit_5metric",
                "metrics_available": used, "metrics_unavailable": missing,
                "manual_anchor": {"b": b, "r": r}}

    scored_path = OUT_DIR / tag / f"{ctx}_scored.csv"
    _run([str(ENV_BIN / "cell-eval2"), "score", "--user-agg", str(agg),
          "--real-bundle", str(bundle), "-o", str(scored_path)])
    scored = pd.read_csv(scored_path).set_index("metric")
    per_metric, missing = {}, []
    for m in METRICS:
        if m not in scored.index:
            missing.append(SHORT[m])
            continue
        v = scored.loc[m, "from_replicate"]
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = float("nan")
        if np.isfinite(v):
            per_metric[SHORT[m]] = v
        else:
            missing.append(SHORT[m])
    try:
        overall = float(scored.loc["avg_score", "from_replicate"])
    except (KeyError, TypeError, ValueError):
        overall = float("nan")
    manual_mean = (float(sum(per_metric.values()) / len(per_metric))
                   if per_metric else float("nan"))
    if not np.isfinite(overall):
        overall = manual_mean
    return {"per_metric": per_metric, "per_metric_raw": per_metric_raw,
            "overall": overall,
            "bundle_mode": "bundle" if not missing else "bundle_partial",
            "metrics_available": list(per_metric.keys()),
            "metrics_unavailable": missing}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", required=True, help="逗号分隔的 local_bench tag（目录名）")
    ap.add_argument("--contexts", default="P_H1,Q_HepG2,R_Jurkat")
    ap.add_argument("--build-bundles", action="store_true",
                    help="bundle 不存在时现场构建（baseline + anchor，较慢）")
    ap.add_argument("--allow-degenerate-baseline", action="store_true",
                    help="baseline 步加 --allow-degenerate-baseline（全深度 HCT116 ref）")
    ap.add_argument("--out", default=str(OUT_DIR / "anchor_normalized.json"))
    args = ap.parse_args()

    contexts = args.contexts.split(",")
    bundles: dict[str, Path | None] = {}
    manuals: dict[str, dict] = {}
    for ctx in contexts:
        if not args.build_bundles:
            b = OUT_DIR / "bundles" / ctx
            bundles[ctx] = b if (b / "manifest.json").exists() else None
            if bundles[ctx] is None:
                manuals[ctx] = manual_anchor(ctx)
            continue
        try:
            bundles[ctx] = build_bundle(ctx, allow_degenerate=args.allow_degenerate_baseline)
        except Exception as e:  # noqa: BLE001
            print(f"[{ctx}] bundle build failed ({e}); → manual anchor 5-metric fallback",
                  flush=True)
            bundles[ctx] = None
            manuals[ctx] = manual_anchor(ctx)

    result: dict = {"scale": "official anchor-normalized (0=baseline, 1=replicate)",
                    "per_tag": {}}
    for tag in args.tags.split(","):
        try:
            per_ctx = {}
            for ctx in contexts:
                per_ctx[ctx] = score_tag_ctx(tag, ctx, bundles[ctx],
                                             None if bundles[ctx] is not None
                                             else manuals.get(ctx))
        except Exception as e:  # noqa: BLE001 — 单 tag 失败不拖垮其余 tag
            print(f"[{tag}] scoring failed: {e}", flush=True)
            result["per_tag"][tag] = {"error": str(e)}
            continue
        avail = [c for c in contexts if per_ctx[c]["per_metric"]]
        overall = {}
        for key in ("per_metric", "per_metric_raw"):
            keys = set.intersection(*[set(per_ctx[c][key]) for c in avail]) if avail else set()
            overall[key] = {k: float(sum(per_ctx[c][key][k] for c in avail) / len(avail))
                            for k in sorted(keys)}
        ov = [per_ctx[c]["overall"] for c in avail]
        overall["overall"] = float(sum(ov) / len(ov)) if ov else float("nan")
        result["per_tag"][tag] = {"per_context": per_ctx, "overall": overall}
        print(f"[{tag}] overall(official)={overall['overall']:.4f} "
              + " ".join(f"{k}={v:.4f}" for k, v in overall["per_metric"].items())
              + (f" [{per_ctx[avail[0]]['bundle_mode']}]" if avail else ""), flush=True)

    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
