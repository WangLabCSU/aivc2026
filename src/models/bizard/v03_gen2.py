"""Bizard v0.3 PBT gen-2: 表征级变异鸟的代际进化（Δ̂ 级插值, 零重训）。

--fitness  8 鸟全局适应度表（已有 bench 复用: 单鸟 Q_HepG2 + 聚合体五场景）
           → workdir/gen2_fitness_table.json
--cross    头 2 鸟 × 交叉/扰动 + 跨骨干杂交 → 4 子代 Δ̂（Q_HepG2 + submit_A/B/C）
           → workdir/gen2/pred_g2_*.parquet + gen2_children.json
--bench    子代/聚合 bench（复用 bench_bird 链, λ2.5 主口径）
--aggregate 4 子代 soft-rank 聚合 → gen2/agg_delta_{ctx}.parquet（+可选 bench）

子代设计（父代 = Q_HepG2 适应度头 2 鸟 xgb_s235 .4146 / xgb_rowboot .4137）:
  g2_c1 crossover55   0.5·P1 + 0.5·P2                       （头鸟等权交叉）
  g2_c2 crossover73   0.7·P1 + 0.3·P2                       （适应度偏置交叉）
  g2_c3 heux235       0.3·heu_dual + 0.7·P1                 （跨骨干杂交, v0.4 建议方向）
  g2_c4 fitw_heu10    heu .10 + P1 .30 + P2 .25 + b3r .20 + sigdrop .15
                                                            （适应度加权池, 弱鸟 w=0.1 只供去相关）

用法:
  python src/models/bizard/v03_gen2.py --config configs/bizard/v03_gen2.yaml --fitness
  python src/models/bizard/v03_gen2.py --config ... --cross
  python src/models/bizard/v03_gen2.py --config ... --bench --tag bz03g2_g2_c1 --contexts Q_HepG2
  python src/models/bizard/v03_gen2.py --config ... --aggregate [--bench-agg --contexts Q_HepG2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
sys.path.insert(0, str(ROOT))

BENCH_OUT = ROOT / "outputs/metrics/local_bench"
LAM = 2.5

# 子代基因型: name → {parent: weight}（Δ̂ 级线性插值, 权重和=1）
CHILDREN: dict[str, dict[str, float]] = {
    "g2_c1": {"xgb_s235": 0.5, "xgb_rowboot": 0.5},
    "g2_c2": {"xgb_s235": 0.7, "xgb_rowboot": 0.3},
    "g2_c3": {"heu_dual": 0.3, "xgb_s235": 0.7},
    "g2_c4": {"heu_dual": 0.10, "xgb_s235": 0.30, "xgb_rowboot": 0.25,
              "v02_b3r": 0.20, "xgb_sigdrop": 0.15},
}


CHILDREN_PARENT_ORDER = ["xgb_s235", "xgb_rowboot", "v02_b3r", "xgb_sigdrop",
                         "mlp_mse", "mlp_s235", "mlp_sigsub", "heu_dual"]


def v1_workdir() -> Path:
    return ROOT / "data/interim/bizard/v03"


def g2_workdir(cfg: dict) -> Path:
    d = ROOT / cfg["workdir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------- fitness
def do_fitness() -> dict:
    """全局适应度: 8 鸟 Q_HepG2 anchor 归一化 overall（已有）+ 聚合体五场景（已有）。"""
    birds = list(CHILDREN_PARENT_ORDER)
    rows = []
    for b in birds:
        p = BENCH_OUT / f"bz03_{b}" / "bizard_summary.json"
        if not p.exists():
            print(f"[fitness] {b}: no summary, skip", flush=True)
            continue
        d = json.loads(p.read_text())
        rows.append({"bird": b, "per_ctx": d["per_ctx"],
                     "fitness": float(d["per_ctx"]["Q_HepG2"])})
    rows.sort(key=lambda r: r["fitness"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    agg = {}
    for tag in ("bz03_agg_lam2.5", "bz03_aggw_lam2.5", "bz01_agg_sr"):
        p = BENCH_OUT / tag / "bizard_summary.json"
        if p.exists():
            d = json.loads(p.read_text())
            pc = d.get("per_ctx") or {k: v["overall"] if isinstance(v, dict) else v
                                      for k, v in (d.get("per_context") or {}).items()}
            agg[tag] = pc
    out = {"rank_key": "Q_HepG2 anchor-normalized overall (lam per tag)",
           "birds": rows, "aggregates": agg,
           "parents": {"P1": "xgb_s235", "P2": "xgb_rowboot"}}
    (v1_workdir() / "gen2_fitness_table.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps([[r["rank"], r["bird"], round(r["fitness"], 4)] for r in rows],
                     indent=1), flush=True)
    return out


# --------------------------------------------------------------------- cross
def _load_grid(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(path)
    return df[["target_gene", "gene"]], df["delta"].to_numpy(np.float32)


def do_cross(cfg: dict) -> None:
    out = g2_workdir(cfg)
    src = v1_workdir()
    ctx_all = ["Q_HepG2", "P_H1", "R_Jurkat", "S_RPE1", "S_HCT116",
               "submit_A", "submit_B", "submit_C"]
    spec = {}
    for name, mix in CHILDREN.items():
        spec[name] = {"mix": mix,
                      "op": ("crossover" if len(mix) == 2 and "heu_dual" not in mix
                             else "backbone_hybrid" if "heu_dual" in mix and len(mix) == 2
                             else "fitness_weighted_pool")}
        for ctx in ctx_all:
            fp = out / f"pred_{name}_{ctx}.parquet"
            if fp.exists():
                print(f"{fp.name} exists, skip", flush=True)
                continue
            t0 = time.time()
            base, acc = None, None
            for parent, w in mix.items():
                g, d = _load_grid(src / f"pred_{parent}_{ctx}.parquet")
                if base is None:
                    base, acc = g, w * d.astype(np.float64)
                else:
                    assert len(g) == len(base) and (g["target_gene"].to_numpy()
                                                    == base["target_gene"].to_numpy()).all() \
                        and (g["gene"].to_numpy() == base["gene"].to_numpy()).all(), \
                        f"{parent}/{ctx} 网格不对齐"
                    acc += w * d.astype(np.float64)
            delta = np.clip(acc, -4.0, 4.0).astype(np.float32)
            pd.DataFrame({"target_gene": base["target_gene"], "gene": base["gene"],
                          "delta": delta}).to_parquet(fp)
            print(f"{fp.name}: rows={len(delta)} absmean={np.abs(delta).mean():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    (out / "gen2_children.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    print(json.dumps(spec, indent=1), flush=True)


# ---------------------------------------------------------------- bench hooks
def bench(tag: str, delta_path: Path, contexts: list[str], lam: float) -> None:
    import src.models.bizard.bench_bird as bb
    outdir = BENCH_OUT / tag
    outdir.mkdir(parents=True, exist_ok=True)
    per_ctx: dict[str, float] = {}
    per_metric: dict[str, dict] = {}
    sm = outdir / "bizard_summary.json"
    if sm.exists():
        prev = json.loads(sm.read_text())
        per_ctx.update(prev.get("per_ctx") or {})
        per_metric.update(prev.get("per_metric") or {})
    for ctx in contexts:
        if ctx in per_ctx:
            print(f"[{tag}/{ctx}] already {per_ctx[ctx]:.4f}, skip", flush=True)
            continue
        t0 = time.time()
        delta = pd.read_parquet(delta_path)
        pp = bb.build_pred(ctx, delta, lam, outdir, seed=0, n_cells=400, max_fold=8.0)
        print(f"[{tag}/{ctx}] pred built ({time.time()-t0:.0f}s)", flush=True)
        bb.run_celleval(ctx, pp, outdir)
        res = bb.score_ctx(tag, ctx)
        per_ctx[ctx] = res["overall"]
        per_metric[ctx] = res.get("per_metric") or {}
        print(f"[{tag}/{ctx}] overall={res['overall']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        bb.save_json({"tag": tag, "lam": lam, "per_ctx": dict(per_ctx),
                      "per_metric": per_metric}, sm)


# ----------------------------------------------------------------- aggregate
def do_aggregate(cfg: dict, contexts: list[str]) -> None:
    import src.models.bizard.swarm as swarm
    out = g2_workdir(cfg)
    names = list(CHILDREN)
    for ctx in contexts:
        fp = out / f"agg_delta_{ctx}.parquet"
        if fp.exists():
            print(f"{fp.name} exists, skip", flush=True)
            continue
        paths = [out / f"pred_{b}_{ctx}.parquet" for b in names]
        have = [n for n, p in zip(names, paths) if p.exists()]
        dfs = [pd.read_parquet(p) for p in paths if p.exists()]
        base = dfs[0][["target_gene", "gene"]]
        preds = []
        for df in dfs:
            assert len(df) == len(base) \
                and (df["target_gene"].to_numpy() == base["target_gene"].to_numpy()).all() \
                and (df["gene"].to_numpy() == base["gene"].to_numpy()).all()
            preds.append(df["delta"].to_numpy(np.float32))
        agg = swarm.aggregate(preds, method=cfg["aggregate"]["method"])
        pd.DataFrame({"target_gene": base["target_gene"], "gene": base["gene"],
                      "delta": agg}).to_parquet(fp)
        print(f"agg {ctx}: birds={have} method={cfg['aggregate']['method']} "
              f"absmean={np.abs(agg).mean():.4f}", flush=True)


# --------------------------------------------------------------------- report
def _pc(path: Path) -> dict:
    d = json.loads(path.read_text())
    return d.get("per_ctx") or {k: v["overall"] if isinstance(v, dict) else v
                                for k, v in (d.get("per_context") or {}).items()}


def do_report(cfg: dict) -> dict:
    """代际表 + 三代聚合对比 → workdir/pbt_report.json + outputs/metrics JSON。"""
    out = g2_workdir(cfg)
    gen2 = []
    for name in CHILDREN:
        p = BENCH_OUT / f"bz03g2_{name}" / "bizard_summary.json"
        if p.exists():
            gen2.append({"bird": name, "mix": CHILDREN[name], "per_ctx": _pc(p)})
    gen2.sort(key=lambda r: -(r["per_ctx"].get("Q_HepG2") or np.nan))
    for i, r in enumerate(gen2):
        r["rank_Q"] = i + 1
        vals = [v for v in r["per_ctx"].values() if np.isfinite(v)]
        r["mean5"] = float(np.mean(vals)) if len(vals) == 5 else None
    g1_fit = json.loads((v1_workdir() / "gen2_fitness_table.json").read_text())
    agg_gen2 = _pc(BENCH_OUT / "bz03g2_agg_lam2.5" / "bizard_summary.json")
    refs = {"v0.1_bz01_agg_sr": _pc(BENCH_OUT / "bz01_agg_sr" / "bizard_summary.json"),
            "gen1_bz03_agg_lam2.5": _pc(BENCH_OUT / "bz03_agg_lam2.5" / "bizard_summary.json"),
            "gen1_weighted_bz03_aggw": _pc(BENCH_OUT / "bz03_aggw_lam2.5" / "bizard_summary.json"),
            "gen2_bz03g2_agg_lam2.5": agg_gen2}
    comp = {k: {"per_ctx": v, "mean5": float(np.mean(list(v.values())))}
            for k, v in refs.items()}
    rep = {"gen": 2, "rank_key": "Q_HepG2 anchor-normalized overall",
           "children": gen2, "gen1_table": g1_fit["birds"],
           "aggregate_comparison": comp,
           "verdict": {"gen2_vs_gen1_Q": round(agg_gen2["Q_HepG2"]
                                               - refs["gen1_bz03_agg_lam2.5"]["Q_HepG2"], 4),
                       "gen2_vs_gen1_mean5": round(comp["gen2_bz03g2_agg_lam2.5"]["mean5"]
                                                   - comp["gen1_bz03_agg_lam2.5"]["mean5"], 4),
                       "gen2_vs_v01_Q": round(agg_gen2["Q_HepG2"]
                                              - refs["v0.1_bz01_agg_sr"]["Q_HepG2"], 4),
                       "gen2_vs_v01_mean5": round(comp["gen2_bz03g2_agg_lam2.5"]["mean5"]
                                                  - comp["v0.1_bz01_agg_sr"]["mean5"], 4),
                       "pbt_promoted": agg_gen2["Q_HepG2"] > refs["gen1_bz03_agg_lam2.5"]["Q_HepG2"]},
           "submission": {"h5ad": "outputs/submissions/20261003_bizard_v03_gen2.h5ad",
                          "vcc": "outputs/submissions/20261003_bizard_v03_gen2.vcc",
                          "lam": 2.5}}
    (out / "pbt_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    metrics = ROOT / "outputs/metrics/20261003_bizard_v03_pbt_gen2.json"
    metrics.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps({"verdict": rep["verdict"],
                      "mean5": {k: round(v["mean5"], 4) for k, v in comp.items()},
                      "children_Q": {r["bird"]: round(r["per_ctx"]["Q_HepG2"], 4)
                                     for r in gen2}}, indent=1, ensure_ascii=False), flush=True)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bizard/v03_gen2.yaml")
    ap.add_argument("--fitness", action="store_true")
    ap.add_argument("--cross", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-agg", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--delta", default=None, help="bench 输入 Δ̂ parquet（默认按 tag 解析）")
    ap.add_argument("--contexts", default="Q_HepG2")
    ap.add_argument("--lam", type=float, default=LAM)
    args = ap.parse_args()
    with open(ROOT / args.config) as f:
        import yaml
        cfg = yaml.safe_load(f)
    ctxs = args.contexts.split(",")
    if args.fitness:
        do_fitness()
    if args.cross:
        do_cross(cfg)
    if args.aggregate:
        do_aggregate(cfg, ctxs)
    if args.bench:
        assert args.tag, "--bench 需要 --tag"
        if args.delta:
            dp = ROOT / args.delta
        else:
            dp = g2_workdir(cfg) / f"pred_{args.tag.replace('bz03g2_', '')}_{ctxs[0]}.parquet"
        bench(args.tag, dp, ctxs, args.lam)
    if args.bench_agg:
        for ctx in ctxs:
            bench(f"bz03g2_agg_lam{args.lam:g}",
                  g2_workdir(cfg) / f"agg_delta_{ctx}.parquet", [ctx], args.lam)
    if args.report:
        do_report(cfg)


if __name__ == "__main__":
    main()
