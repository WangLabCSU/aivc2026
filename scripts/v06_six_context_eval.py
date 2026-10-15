"""Six-context fresh eval of the frozen v0.5 delta (gate-lock option_b path).

Gate-lock v06_gate_lock_2026-10-15, six_context_handling.choice = "option_b":
evaluate the v0.5 delta fresh on the six gate contexts
(gwps / k562_ess / rpe1 / jurkat / hepg2 / hct116) with the current local bench
pipeline, produce a new artifact, freeze its sha256 into the gate lock as the
append-only field six_context_artifact.
Gates read from this artifact: 门2 = fid 非负 >= 5/6; 门3 = pds 回退 <= 0.01.

Delta sources (option a default; --delta-source submit switches the two new
contexts to the literal submit delta -> degenerate all-NTC-copy eval):
  4 reused bench panels (终选A frozen chain, read as-is, no re-clip).
  Vintage provenance (bit-exact rebuild proof, 2026-10-15 archaeology): the
  frozen bz04_final_c3s / bz03g2_agg preds were built from the gen2/ vintage
  (data/interim/bizard/v03/gen2/, Oct 6) — rebuilt-from-gen2 preds match the
  frozen preds with max diff 0.0; the top-level v03/agg_delta_{ctx}.parquet
  files (Oct 4) are an older vintage and do NOT reproduce them.
    Q_HepG2   -> v03/gen2/agg_delta_Q_HepG2.parquet
    R_Jurkat  -> v03/gen2/agg_delta_R_Jurkat.parquet
    S_RPE1    -> v03/gen2/agg_delta_S_RPE1.parquet
    S_HCT116  -> v03/gen2/pred_g2_c3_S_HCT116.parquet (g2_c3 single bird;
                 bz04_final_c3s used the c3 bird for S_HCT116 only)
  new contexts (option a):
    gwps, k562_ess -> v03/heu_agg_gwps.parquet (logfc->delta rename, clip ±4.0;
                     exact K562 source table of the v0.5 heuristic bird —
                     v03_run.heuristic_delta semantics, v0.5-lineage-legitimate)
  P_H1 (non-gate) -> gen2/agg_delta_P_H1.parquet: fresh re-eval used only as a
    driver correctness self-check vs the frozen per_metric table.

Mechanics: calls bench_bird.build_pred / run_celleval / score_ctx directly and
NEVER bench_bird.main(), bypassing the SCENARIO_ORDER assert; bench_bird.load_cfg
(module-level binding) is monkeypatched to inject scenarios gwps/k562_ess —
no committed config file is touched.

--check: read-only validation (delta mapping, load_cfg patch, bundle manifests,
panel coverage counts, frozen gate values); produces no artifact.
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

import src.models.bizard.bench_bird as bench_bird  # noqa: E402
from src.models.bizard.swarm import load_cfg  # noqa: E402

V03 = ROOT / "data/interim/bizard/v03"
# Frozen table assembles from two summaries: bz04_final_c3s holds per_metric for
# P_H1/R_Jurkat/S_RPE1/S_HCT116 but reused Q_HepG2's overall from bz03g2_agg
# (note_Q: same delta, deterministic chain) — Q's per_metric lives only there.
FROZEN_SUMMARY = ROOT / "outputs/metrics/local_bench/bz04_final_c3s/bizard_summary.json"
FROZEN_Q_SUMMARY = ROOT / "outputs/metrics/local_bench/bz03g2_agg_lam2.5/bizard_summary.json"

GATE_CTXS = ["gwps", "k562_ess", "S_RPE1", "R_Jurkat", "Q_HepG2", "S_HCT116"]
NEW_CTXS = ["gwps", "k562_ess"]
REUSED = ["S_RPE1", "R_Jurkat", "Q_HepG2", "S_HCT116"]  # gate contexts with frozen baselines
SELF_CHECK_CTX = "P_H1"  # non-gate, fresh re-eval vs frozen only
# Run order: self-check + reused panels first — a deviation-vs-frozen abort must
# happen BEFORE any new-context artifact exists on disk.
RUN_ORDER = [SELF_CHECK_CTX] + REUSED + NEW_CTXS

DELTA_HEUR = {
    "Q_HepG2": V03 / "gen2/agg_delta_Q_HepG2.parquet",
    "R_Jurkat": V03 / "gen2/agg_delta_R_Jurkat.parquet",
    "S_RPE1": V03 / "gen2/agg_delta_S_RPE1.parquet",
    "S_HCT116": V03 / "gen2/pred_g2_c3_S_HCT116.parquet",
    "P_H1": V03 / "gen2/agg_delta_P_H1.parquet",
    "gwps": V03 / "heu_agg_gwps.parquet",
    "k562_ess": V03 / "heu_agg_gwps.parquet",
}
DELTA_SUBMIT = dict(DELTA_HEUR)
DELTA_SUBMIT["gwps"] = V03 / "gen2/pred_g2_c3_submit_A.parquet"
DELTA_SUBMIT["k562_ess"] = V03 / "gen2/pred_g2_c3_submit_A.parquet"


def patch_load_cfg() -> None:
    """Inject scenarios gwps/k562_ess into bench_bird's load_cfg (module binding)."""
    orig = bench_bird.load_cfg

    def patched(path=None):
        cfg = orig(path)
        cfg["scenarios"].setdefault("gwps", {"train_ctx": "gwps", "control": "non-targeting"})
        cfg["scenarios"].setdefault("k562_ess", {"train_ctx": "k562_ess", "control": "non-targeting"})
        return cfg

    bench_bird.load_cfg = patched


def load_delta(path: Path, heu: bool) -> pd.DataFrame:
    cols = ["target_gene", "gene", "logfc" if heu else "delta"]
    d = pd.read_parquet(path, columns=cols)
    if heu:
        d = d.rename(columns={"logfc": "delta"})
        d["delta"] = d["delta"].clip(-4.0, 4.0)  # v03_run.heuristic_delta max_abs_delta
    assert not d.duplicated(["target_gene", "gene"]).any(), f"duplicate rows in {path}"
    return d


def frozen_table() -> dict:
    """Return {ctx: {overall, per_metric}} assembled from the two frozen summaries."""
    b4 = json.loads(FROZEN_SUMMARY.read_text())
    q = json.loads(FROZEN_Q_SUMMARY.read_text())
    tab = {}
    for ctx, ov in b4["per_ctx"].items():
        pm = b4["per_metric"].get(ctx)
        if pm is None and ctx == "Q_HepG2":
            pm = q["per_metric"]["Q_HepG2"]
        tab[ctx] = {"overall": ov, "per_metric": pm}
    return tab


def check(lam: float) -> None:
    patch_load_cfg()
    cfg = bench_bird.load_cfg()
    print("scenarios keys:", sorted(cfg["scenarios"].keys()))
    print("gwps ccfg:", cfg["scenarios"]["gwps"])
    print("k562_ess ccfg:", cfg["scenarios"]["k562_ess"])

    frozen = frozen_table()
    print("\nfrozen (bz04_final_c3s) gate values:")
    for ctx in REUSED:
        pm = frozen[ctx]["per_metric"]
        print(f"  {ctx:9s} overall={frozen[ctx]['overall']:.6f} "
              f"fid={pm['fid']:.4f} pds={pm['pds']:.4f}")

    print("\ncoverage vs panel bench targets:")
    for ctx, dp in DELTA_HEUR.items():
        targets = set(json.loads((bench_bird.BENCH / ctx / "targets.json").read_text()))
        dtg = set(pd.read_parquet(dp, columns=["target_gene"]).target_gene.unique())
        print(f"  {ctx:9s} bench={len(targets):4d} delta_targets={len(dtg):5d} "
              f"hit={len(targets & dtg):4d} ({len(targets & dtg) / len(targets):6.1%})")

    for ctx in GATE_CTXS + [SELF_CHECK_CTX]:
        b = bench_bird.OUT_DIR / "bundles" / ctx
        print(f"bundle {ctx}: manifest={'OK' if (b / 'manifest.json').exists() else 'MISSING'}")


def run(lam: float, tag: str, delta_source: str, force: bool) -> dict:
    patch_load_cfg()
    cfg = bench_bird.load_cfg()
    dmap = DELTA_HEUR if delta_source == "heur" else DELTA_SUBMIT
    outdir = bench_bird.OUT_DIR / tag
    outdir.mkdir(parents=True, exist_ok=True)

    frozen = frozen_table()
    per_ctx: dict[str, dict] = {}
    selfcheck_diffs: dict[str, float] = {}
    for ctx in RUN_ORDER:
        t0 = time.time()
        pp = outdir / f"{ctx}_pred.h5ad"
        if not pp.exists():
            delta = load_delta(dmap[ctx], heu=ctx in NEW_CTXS and delta_source == "heur")
            pp = bench_bird.build_pred(ctx, delta, lam, outdir, seed=0,
                                       n_cells=cfg["predict"]["n_cells_per_pert"],
                                       max_fold=cfg["predict"]["max_fold"])
            print(f"[{ctx}] pred built ({time.time()-t0:.0f}s)", flush=True)
        else:
            print(f"[{ctx}] pred cached", flush=True)
        edir = outdir / f"{ctx}_eval"
        if not (edir / "agg_results.csv").exists():
            bench_bird.run_celleval(ctx, pp, outdir)
        res = bench_bird.score_ctx(tag, ctx)
        per_ctx[ctx] = res
        print(f"[{ctx}] overall={res['overall']:.6f} mode={res.get('bundle_mode')}", flush=True)

        if ctx in frozen:
            diffs = {m: abs(res["per_metric"][m] - frozen[ctx]["per_metric"][m])
                     for m in frozen[ctx]["per_metric"] if m in res["per_metric"]}
            md = max(diffs.values()) if diffs else float("inf")
            od = abs(res["overall"] - frozen[ctx]["overall"])
            selfcheck_diffs[ctx] = max(md, od)
            print(f"[{ctx}] self-check vs frozen: max_metric_diff={md:.2e} overall_diff={od:.2e}")
            if ctx != SELF_CHECK_CTX and md > 1e-4 and not force:
                raise SystemExit(f"{ctx}: fresh re-eval deviates from frozen table "
                                 f"({md:.3e}) — aborting before new contexts; use --force to override")
    gate = {c: per_ctx[c] for c in GATE_CTXS}
    fid_nonneg = [c for c in GATE_CTXS
                  if np.isfinite(gate[c]["per_metric"].get("fid", np.nan))
                  and gate[c]["per_metric"]["fid"] >= 0]
    gate2 = {"rule": "fid 非负 >= 5/6",
             "fid_nonneg": fid_nonneg,
             "count": len(fid_nonneg),
             "pass": len(fid_nonneg) >= 5}
    pds_reg = {}
    for ctx in REUSED:
        fp = frozen[ctx]["per_metric"]["pds"]
        npd = gate[ctx]["per_metric"].get("pds", np.nan)
        pds_reg[ctx] = {"frozen": fp, "fresh": npd, "regress": npd - fp}
    gate3 = {"rule": "pds 回退 <= 0.01 (4 reused contexts)",
             "per_ctx": pds_reg,
             "pass": all(np.isfinite(v["regress"]) and v["regress"] <= 0.01 for v in pds_reg.values()),
             "note": "gwps/k562_ess have no frozen pds baseline; reported absolute only"}
    new_pds = {c: gate[c]["per_metric"].get("pds", np.nan) for c in NEW_CTXS}

    vals = [v["overall"] for v in gate.values() if np.isfinite(v.get("overall", np.nan))]
    summary = {"tag": tag, "lam": lam, "delta_source": delta_source,
               "seed": 0, "n_cells": cfg["predict"]["n_cells_per_pert"],
               "max_fold": cfg["predict"]["max_fold"],
               "gate_contexts": GATE_CTXS,
               "per_ctx_overall": {c: v["overall"] for c, v in gate.items()},
               "mean6_overall": float(np.mean(vals)) if len(vals) == 6 else float("nan"),
               "per_metric": {c: v["per_metric"] for c, v in gate.items()},
               "bundle_mode": {c: v.get("bundle_mode") for c, v in gate.items()},
               "self_check_vs_frozen": selfcheck_diffs,
               "p_h1_selfcheck_overall": per_ctx[SELF_CHECK_CTX]["overall"],
               "gate2": gate2, "gate3": gate3,
               "new_context_pds_absolute": new_pds}
    bench_bird.save_json(summary, outdir / "bizard_summary.json")
    print(json.dumps({"gate2": gate2["count"], "gate2_pass": gate2["pass"],
                      "gate3_pass": gate3["pass"], "mean6": summary["mean6_overall"]},
                     ensure_ascii=False, indent=1), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=2.5)
    ap.add_argument("--tag", default="sixctx_v05")
    ap.add_argument("--delta-source", choices=["heur", "submit"], default="heur")
    ap.add_argument("--force", action="store_true",
                    help="do not abort on self-check deviation vs frozen table")
    ap.add_argument("--check", action="store_true", help="read-only validation, no artifact")
    args = ap.parse_args()
    if args.check:
        check(args.lam)
    else:
        run(args.lam, args.tag, args.delta_source, args.force)
