"""Bizard 报告: 鸟群训练矩阵（每鸟配置 + 局部/全局适应度）+ 聚合 vs 单鸟 vs 基线对比表。

读取 outputs/metrics/local_bench/bz01_*/bizard_summary.json →
  outputs/metrics/20261003_bizard_v01_swarm.json（机读）
  docs/bizard/v01_results.md 的表格段（stdout markdown）

用法: python src/models/bizard/report.py [--out-md docs/bizard/v01_results.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.bizard.swarm import (  # noqa: E402
    ROOT, SCENARIO_ORDER, birds, global_fitness, load_cfg)

OUT_DIR = ROOT / "outputs/metrics/local_bench"

# 基线（bench Q_HepG2 bundle 归一化 / 官方）:
BASELINES = {
    "dualsrc-deep λ3.5 (现行最优)": {"official": 0.0506, "Q_HepG2": 0.3182},
    "fusion v3.5 s1 LOCO λ2.5": {"official": None, "Q_HepG2": 0.3313},
    "heuristic λ3.5 (affine 拟合点)": {"official": 0.0247, "Q_HepG2": 0.2871},
}


def load_summary(tag: str) -> dict | None:
    p = OUT_DIR / tag / "bizard_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default="outputs/metrics/20261003_bizard_v01_swarm.json")
    args = ap.parse_args()
    cfg = load_cfg()
    fit = cfg["fitness"]
    rows, agg_rows = [], []

    for b in birds(cfg):
        tag = f"bz01_{b['name']}"
        s = load_summary(tag)
        row = {"bird": b["name"], "scenario": b["scenario"], "main_weight": b["main_weight"],
               "seed": b["seed"], "aux_weight": b["aux_weight"], "reuse": bool(b.get("reuse")),
               "tag": tag, "per_ctx": {}, "lam": None}
        if s:
            row["per_ctx"] = {c: v["overall"] for c, v in s["per_context"].items()}
            row["lam"] = s["lam"]
            gf = global_fitness(row["per_ctx"], fit)
            row["local_fitness"] = (row["per_ctx"].get(b["scenario"])
                                    if b["scenario"] else None)
            row["global"] = gf
        rows.append(row)

    for tag_prefix in ["bz01_agg_sr", "bz01_agg_simple", "bz01_agg_lam3.5"]:
        s = load_summary(tag_prefix)
        if s:
            per = {c: v["overall"] for c, v in s["per_context"].items()}
            agg_rows.append({"tag": tag_prefix, "lam": s["lam"], "per_ctx": per,
                             "global": global_fitness(per, fit),
                             "mean_local": s["mean_overall"]})

    report = {"version": "bizard-v0.1", "fitness_rule": {
        "local": "主场景 anchor-normalized overall",
        "global": "clip(2.0958·local_mean − 0.5259, −0.11, +0.10) − 0.15·max(0, −min_ctx)",
        "note": "affine 为 heuristic 家族标定, 同族相对比较用; 鸟覆盖=主场景+Q_HepG2 (partial)"},
        "birds": rows, "aggregates": agg_rows, "baselines": BASELINES}
    outp = ROOT / args.out_json
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {outp}")

    md = ["## 鸟群训练矩阵", "",
          "| 鸟 | 主场景 | τ(主行权) | seed | aux | 局部适应度 | 全局适应度(affine均值) | 场景min | P_H1 | Q | R | S_RPE1 | S_HCT116 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = r.get("global") or {}
        pc = r["per_ctx"]
        md.append("| {bird} | {sc} | {mw} | {sd} | {aux} | {loc} | {g} | {mn} | {p} | {q} | {rr} | {s1} | {s2} |".format(
            bird=r["bird"], sc=r["scenario"] or "—", mw=r["main_weight"], sd=r["seed"],
            aux=r["aux_weight"],
            loc=f"{r['local_fitness']:.4f}" if r.get("local_fitness") else "—",
            g=f"{g.get('affine_mean_raw', float('nan')):.4f}" if g else "—",
            mn=f"{g.get('min_ctx', float('nan')):.4f}" if g else "—",
            p=f"{pc['P_H1']:.4f}" if "P_H1" in pc else "—",
            q=f"{pc['Q_HepG2']:.4f}" if "Q_HepG2" in pc else "—",
            rr=f"{pc['R_Jurkat']:.4f}" if "R_Jurkat" in pc else "—",
            s1=f"{pc['S_RPE1']:.4f}" if "S_RPE1" in pc else "—",
            s2=f"{pc['S_HCT116']:.4f}" if "S_HCT116" in pc else "—"))
    md += ["", "## 群飞聚合 vs 单鸟 vs 基线（五场景 anchor 归一化; 括号=affine 官方预期）", "",
           "| 实体 | λ | P_H1 | Q_HepG2 | R_Jurkat | S_RPE1 | S_HCT116 | 局部均值 | 全局适应度 |", "|---|---|---|---|---|---|---|---|---|"]
    for a in agg_rows:
        pc = a["per_ctx"]
        cells = {k: (f"{pc[k]:.4f}" if k in pc else "—") for k in
                 ["P_H1", "Q_HepG2", "R_Jurkat", "S_RPE1", "S_HCT116"]}
        md.append("| {t} | {lam} | {p} | {q} | {r} | {s1} | {s2} | {m:.4f} | {g:.4f} |".format(
            t=a["tag"], lam=a["lam"], p=cells["P_H1"], q=cells["Q_HepG2"],
            r=cells["R_Jurkat"], s1=cells["S_RPE1"], s2=cells["S_HCT116"],
            m=a["mean_local"], g=a["global"]["affine_mean_raw"]))
    print("\n".join(md))
    if args.out_md:
        p = ROOT / args.out_md
        p.parent.mkdir(parents=True, exist_ok=True)
        existing = p.read_text() if p.exists() else ""
        p.write_text(existing + "\n" + "\n".join(md) + "\n")
        print(f"appended → {p}")


if __name__ == "__main__":
    main()
