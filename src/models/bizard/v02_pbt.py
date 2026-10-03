"""Bizard v0.2 真 PBT: 代际选择（头鸟/末位鸟）+ 高斯扰动配置生成 + 代际适应度表 + 聚合。

--select  读 gen1 各鸟 summary → rank_raw 排名 → 头鸟 T/末位 2 鸟 L1,L2 →
          写 configs/bizard/v02_gen2.yaml（L 继承 T 配置 + 扰动: aux+N(0,0.10),
          main×exp(N(0,0.15)), seed=L.seed+10, 场景/TopK/模块开关继承 T）
--report  汇总 gen1+gen2 适应度表 → workdir/pbt_report.json;
          全体已评鸟 soft-rank 聚合 Q_HepG2 Δ̂ → λ2.5 bench（tag bz02_agg_v02）
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
sys.path.insert(0, str(ROOT))

GEN2_CFG = ROOT / "configs/bizard/v02_gen2.yaml"


def load_summaries(swarm, cfg: dict) -> list[dict]:
    out = []
    for b in swarm.birds(cfg):
        p = swarm.workdir(cfg) / b["name"] / "fitness.json"
        s = ROOT / cfg["bench_out"] / f"bz02_{b['name']}" / "bizard_summary.json"
        if s.exists():
            d = json.loads(s.read_text())
            d["cfg"] = b
            out.append(d)
    return out


def select(cfg_path: str) -> None:
    import src.models.bizard.swarm as swarm
    import yaml
    swarm.CFG_PATH = ROOT / cfg_path
    cfg = swarm.load_cfg()
    fits = load_summaries(swarm, cfg)
    assert len(fits) >= 3, f"gen1 summaries insufficient: {len(fits)}"
    fits.sort(key=lambda d: d["rank_raw"], reverse=True)
    top, losers = fits[0], fits[-2:]
    pert = cfg["pbt"]["perturb"]
    lo, hi = cfg["pbt"]["aux_bounds"]
    gen2 = []
    for L in losers:
        rng = np.random.default_rng(100 + int(L["cfg"]["seed"]))
        aux = float(np.clip(top["cfg"]["aux_weight"] + rng.normal(0, pert["aux_sigma"]), lo, hi))
        main = float(top["cfg"]["main_weight"] * np.exp(rng.normal(0, pert["main_sigma"])))
        gen2.append({"name": f"{L['cfg']['name']}_g2", "scenario": top["cfg"]["scenario"],
                     "main_weight": round(main, 3), "seed": int(L["cfg"]["seed"]) + pert["seed_offset"],
                     "aux_weight": round(aux, 3), "pred_topk": top["cfg"].get("pred_topk"),
                     "module_off": list(top["cfg"].get("module_off") or []),
                     "inherit_from": top["cfg"]["name"], "replaces": L["cfg"]["name"]})
    g2 = copy.deepcopy(cfg)
    g2["birds"] = gen2
    with open(GEN2_CFG, "w") as f:
        yaml.safe_dump(g2, f, allow_unicode=True, sort_keys=False)
    table = [{"rank": i + 1, "bird": d["bird"], "cfg": d["cfg"], "per_ctx": d["per_ctx"],
              "rank_raw": d["rank_raw"], "global": d["global"]} for i, d in enumerate(fits)]
    (swarm.workdir(cfg) / "pbt_gen1_table.json").write_text(
        json.dumps(table, indent=2, ensure_ascii=False))
    print(json.dumps({"top": top["bird"], "losers": [l["bird"] for l in losers],
                      "gen2": gen2}, indent=1, ensure_ascii=False), flush=True)


def report(cfg_path: str) -> None:
    import src.models.bizard.bench_bird as bb
    import src.models.bizard.swarm as swarm
    import yaml
    swarm.CFG_PATH = ROOT / cfg_path
    cfg = swarm.load_cfg()
    gen1 = json.loads((swarm.workdir(cfg) / "pbt_gen1_table.json").read_text())
    gen2_fits = load_summaries(swarm, cfg)
    gen2 = sorted(gen2_fits, key=lambda d: d["rank_raw"], reverse=True)
    # 代际判据: 变异鸟 rank_raw 是否超过被替换鸟（继承有效性）
    inherit = []
    for d in gen2:
        rep = next((r for r in gen1 if r["bird"] == d["cfg"]["replaces"]), None)
        inh = d["cfg"].get("inherit_from")
        src = next((r for r in gen1 if r["bird"] == inh), None)
        inherit.append({"bird": d["bird"], "replaces": d["cfg"]["replaces"],
                        "rank_raw": d["rank_raw"], "replaced_rank_raw": rep["rank_raw"] if rep else None,
                        "delta_vs_replaced": (d["rank_raw"] - rep["rank_raw"]) if rep else None,
                        "inherited_from": inh,
                        "delta_vs_parent": (d["rank_raw"] - src["rank_raw"]) if src else None})
    # 聚合 = PBT 第 2 代种群: gen1 幸存 4 鸟 + gen2 变异鸟（代际替换语义）
    replaced = {d["cfg"]["replaces"] for d in gen2 if d["cfg"].get("replaces")}
    pool = [r["bird"] for r in gen1 if r["bird"] not in replaced]
    pool += [d["bird"] for d in gen2]
    print(f"[pbt] aggregate pool: {pool}", flush=True)
    out = swarm.workdir(cfg)
    agg_tag = "bz02_agg_v02"
    res_agg = {}
    for ctx in ["Q_HepG2"]:
        # 稠密参照网格 = 池中行数最多者（topk 变异鸟行数少 → 左连补 0）
        grid, preds = None, []
        for b in pool:
            df = pd.read_parquet(out / f"pred_{b}_{ctx}.parquet")
            if grid is None or len(df) > len(grid):
                grid = df[["target_gene", "gene"]]
        for b in pool:
            df = pd.read_parquet(out / f"pred_{b}_{ctx}.parquet")
            g = grid.merge(df, on=["target_gene", "gene"], how="left")
            preds.append(np.nan_to_num(g["delta"].to_numpy(np.float32), nan=0.0))
        agg = swarm.aggregate(preds, method=cfg["aggregate"]["method"])
        pd.DataFrame({"target_gene": grid["target_gene"], "gene": grid["gene"],
                      "delta": agg}).to_parquet(out / f"agg_delta_{ctx}.parquet")
        delta = pd.read_parquet(out / f"agg_delta_{ctx}.parquet")
        outdir = ROOT / cfg["bench_out"] / agg_tag
        outdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        pp = bb.build_pred(ctx, delta, cfg["aggregate"]["bench_lambda"], outdir, seed=0,
                           n_cells=cfg["predict"]["n_cells_per_pert"],
                           max_fold=cfg["predict"]["max_fold"])
        bb.run_celleval(ctx, pp, outdir)
        res_agg[ctx] = bb.score_ctx(agg_tag, ctx)
        print(f"[pbt agg/{ctx}] overall={res_agg[ctx]['overall']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    rep = {"gen": 2, "rank_key": cfg["pbt"]["rank_key"],
           "gen1_table": gen1, "gen2_table": [
               {"bird": d["bird"], "cfg": d["cfg"], "per_ctx": d["per_ctx"],
                "rank_raw": d["rank_raw"], "global": d["global"]} for d in gen2],
           "inherit_check": inherit, "aggregate_pool": pool,
           "agg_bench": {c: {"overall": v["overall"], "per_metric": v["per_metric"]}
                         for c, v in res_agg.items()},
           "v01_ref": {"agg_Q_lam2.5": 0.4110, "mean5_lam2.5": 0.2550}}
    swarm.save_json(rep, ROOT / "outputs/metrics/20261003_bizard_v02_pbt.json")
    print(json.dumps(rep["inherit_check"], indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bizard/v02.yaml")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.select:
        select(args.config)
    if args.report:
        report(args.config)
