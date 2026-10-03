"""Bizard L3 群飞聚合: 六鸟 Δ̂ → soft-rank average → 聚合 Δ̂（bench 场景 + A/B/C）。

用法: python src/models/bizard/aggregate.py --bench P_H1,Q_HepG2,R_Jurkat,S_RPE1,S_HCT116 \
        [--submit] [--method soft_rank]
产出: <workdir>/agg_delta_{ctx}.parquet, <workdir>/agg_delta_submit_{A|B|C}.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.bizard.swarm import ROOT, birds, load_cfg, workdir  # noqa: E402


def load_bird_deltas(paths: list) -> pd.DataFrame:
    dfs = [pd.read_parquet(p) for p in paths]
    base = dfs[0][["target_gene", "gene"]]
    deltas = []
    for df in dfs:
        assert (df["target_gene"].to_numpy() == base["target_gene"].to_numpy()).all() \
            and (df["gene"].to_numpy() == base["gene"].to_numpy()).all(), "鸟间网格不对齐"
        deltas.append(df["delta"].to_numpy(np.float32))
    return base, deltas


def aggregate_ctx(cfg: dict, delta_paths: list, out_path, method: str) -> dict:
    if out_path.exists():
        print(f"{out_path.name} exists, skip", flush=True)
        return {}
    t0 = time.time()
    base, deltas = load_bird_deltas(delta_paths)
    agg = aggregate_py(deltas, method)
    pd.DataFrame({"target_gene": base["target_gene"], "gene": base["gene"],
                  "delta": agg}).to_parquet(out_path)
    st = {"n_birds": len(deltas), "method": method,
          "absmean_bird_mean": float(np.mean([np.abs(d).mean() for d in deltas])),
          "absmean_agg": float(np.abs(agg).mean()),
          "pearson_vs_bird_mean": float(np.corrcoef(
              np.mean(deltas, axis=0), agg)[0, 1]),
          "seconds": round(time.time() - t0, 1)}
    print(f"{out_path.name}: {st}", flush=True)
    return st


def aggregate_py(deltas, method):
    from src.models.bizard.swarm import aggregate
    return aggregate(deltas, method=method)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--method", default=None)
    args = ap.parse_args()
    cfg = load_cfg()
    method = args.method or cfg["aggregate"]["method"]
    out = workdir(cfg)
    report: dict = {"method": method, "bench": {}, "submit": {}}

    bl = [b for b in birds(cfg)]
    for ctx in filter(None, args.bench.split(",")):
        paths = [out / f"pred_{b['name']}_{ctx}.parquet" for b in bl]
        miss = [p.name for p in paths if not p.exists()]
        if miss:
            print(f"[{ctx}] 缺鸟 pred {len(miss)}: {miss} — 用其余鸟聚合", flush=True)
            paths = [p for p in paths if p.exists()]
        report["bench"][ctx] = aggregate_ctx(cfg, paths, out / f"agg_delta_{ctx}.parquet",
                                             method)
    if args.submit:
        for cx in ["A", "B", "C"]:
            paths = [out / f"pred_{b['name']}_submit_{cx}.parquet" for b in bl]
            miss = [p.name for p in paths if not p.exists()]
            if miss:
                print(f"[submit {cx}] 缺鸟 pred {len(miss)}: {miss} — 用其余鸟聚合", flush=True)
                paths = [p for p in paths if p.exists()]
            report["submit"][cx] = aggregate_ctx(cfg, paths,
                                                 out / f"agg_delta_submit_{cx}.parquet",
                                                 method)
    (out / "aggregate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
