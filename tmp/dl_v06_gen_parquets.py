#!/usr/bin/env python
"""v0.6 gate4 DL delta generator (CPU/GPU inference, one shot per context).

Loads the live v0.6 run checkpoint (state_v06_gwps_rpe1) and produces per-context
delta parquets schema-isomorphic to data/interim/bizard/v03/gen2/pred_g2_c3_submit_{cx}.parquet
(columns target_gene:str, gene:str, delta:float; row grid identical to the v03
agg_delta chain -> soft_rank_average row-alignment constraint satisfied).

Pipeline per context (PI Option a semantics, frozen):
  1. NTC pool raw counts from the context h5ad
     (submit A/B/C: data/raw/controls/context_{A,B,C}.h5ad, control 'non-targeting';
      bench 5 ctx: data/interim/local_bench/{ctx}/ref.h5ad,
      control = load_cfg()["scenarios"][ctx]["control"]).
  2. normalize_total(11940)+log1p in CONTEXT space, then project to the live
     run's 8192 HVG order (missing genes zero-filled, order immutable) --
     identical to scripts/v06_select_encoder_init.py load_ctx_cells.
  3. basal = NTC pool mean row in 8192 log1p space (ntc_mean caliber).
  4. Per target: official _infer.py virtual-experiment loop -- windows of
     cell_set_len=64 ctrl cells sampled with replacement (rng per ctx, seed 0),
     prepare_batch contract (pert onehot from pert_onehot_map.pt; unseen pert
     -> 'non-targeting' onehot fallback, official semantics; batch = zeros ->
     index 0, official missing-batch fallback) -> model.predict_step(padded=False)
     -> absolute 8192-dim log1p preds, clipped to [0, 14] (official clip_array
     on absolute outputs) -- then mean over sampled cells.
  5. delta8192 = mean_pred(pert) - basal; projected to the context gene grid
     BY NAME via tmp/mapping_run_v06.csv membership (hvg gene -> its 8192
     delta; out-of-set gene -> 0.0). Units: /ln(2) -> log2FC, matching the v03
     delta lineage (v03_run.py logfc->delta rename) consumed by bench_bird
     f = 2^(lam*delta). NO lambda applied here.
  6. Row-order hard assert against the v03 reference parquet, then write
     OUT_DIR/pred_dl_v06_{ctx}.parquet.

Smoke mode (--smoke): CPU, 1 window, first 2 targets per context, output to
tmp/v06_dl_smoke/ -- validates the full chain end to end without the GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
import zlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import LlamaConfig

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")

# LlamaConfig bypass BEFORE importing state / torch.load of the legacy ckpt
# (pattern proven in tmp/run_v06_train.py:15-18).
_orig_validate = LlamaConfig.validate
LlamaConfig.validate = classmethod(lambda cls, self=None: None)
sys.path.insert(0, str(ROOT))
from state.tx.models.state_transition import StateTransitionPerturbationModel  # noqa: E402

RUN_DIR = ROOT / "data/interim/dl_v06/run_v06/state_v06_gwps_rpe1"
# SECURITY: pickle.load / torch.load(weights_only=False) / load_from_checkpoint
# below touch ONLY repo-local, provenance-audited artifacts written by our own
# v0.6 run (var_dims.pkl, pert_onehot_map.pt, checkpoints/best.ckpt) — same
# policy as scripts/v06_select_encoder_init.py; no untrusted sources, no eval.
CKPT = RUN_DIR / "checkpoints/best.ckpt"
MAPPING_CSV = ROOT / "tmp/mapping_run_v06.csv"
V03 = ROOT / "data/interim/bizard/v03"
OUT_DIR = ROOT / "data/interim/bizard/v06_dl"
BENCH = ROOT / "data/interim/local_bench"

NORM_TARGET_SUM = 11940.0     # iter019-proven State input normalization
CELL_SET_LEN = 64             # official recipe (cell_set_len)
SEED = 0                      # ctrl sampling seed (bench_bird build_pred caliber)
CLIP_LO, CLIP_HI = 0.0, 14.0  # official _infer.py clip_array on absolute preds
LN2 = float(np.log(2.0))

SUBMIT_CTX = {  # ctx -> (ctrl h5ad, v03 reference parquet)
    "submit_A": (ROOT / "data/raw/controls/context_A.h5ad", V03 / "agg_delta_submit_A.parquet"),
    "submit_B": (ROOT / "data/raw/controls/context_B.h5ad", V03 / "agg_delta_submit_B.parquet"),
    "submit_C": (ROOT / "data/raw/controls/context_C.h5ad", V03 / "agg_delta_submit_C.parquet"),
}
BENCH_CTX = {  # ctx -> (ctrl h5ad, v03 reference parquet); control from contexts.yaml
    "P_H1":    (BENCH / "P_H1/ref.h5ad",    V03 / "agg_delta_P_H1.parquet"),
    "Q_HepG2": (BENCH / "Q_HepG2/ref.h5ad", V03 / "agg_delta_Q_HepG2.parquet"),
    "R_Jurkat": (BENCH / "R_Jurkat/ref.h5ad", V03 / "agg_delta_R_Jurkat.parquet"),
    "S_RPE1":  (BENCH / "S_RPE1/ref.h5ad",  V03 / "agg_delta_S_RPE1.parquet"),
    "S_HCT116": (BENCH / "S_HCT116/ref.h5ad", V03 / "agg_delta_S_HCT116.parquet"),
}


def read_var_names(v):
    for key in ("gene_name", "_index"):
        if key not in v:
            continue
        o = v[key]
        if isinstance(o, h5py.Dataset):
            arr = o[:]
        elif isinstance(o, h5py.Group) and "categories" in o:  # categorical
            arr = o["categories"][:]
        elif isinstance(o, h5py.Group) and "values" in o:  # nullable-string-array
            vals, mask = o["values"][:], o["mask"][:]
            return [x.decode() if isinstance(x, bytes) else str(x) for x in vals]
        else:
            continue
        return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]
    return None


def read_obs_str(f, key):
    o = f["obs"][key]
    if isinstance(o, h5py.Dataset):
        arr = o[:]
        return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]
    cats = [x.decode() if isinstance(x, bytes) else str(x) for x in o["categories"][:]]
    codes = o["codes"][:]
    return [cats[int(c)] if c >= 0 else None for c in codes]


def read_csr_rows(f, row_ids):
    """Selected rows of an h5ad X (csr/csc/dense) as dense f32 (frozen Phase-2 helper)."""
    X = f["X"]
    if isinstance(X, h5py.Dataset):
        return np.asarray(X[row_ids], dtype=np.float32)
    n, g = (int(x) for x in X.attrs["shape"])
    enc = X.attrs.get("encoding-type", "csr_matrix")
    out = np.zeros((len(row_ids), g), np.float32)
    if enc.startswith("csr"):
        indptr = X["indptr"][:]
        pos = {r: i for i, r in enumerate(row_ids)}
        for r in row_ids:
            a, b = int(indptr[r]), int(indptr[r + 1])
            out[pos[r], X["indices"][a:b]] = X["data"][a:b]
    else:  # csc
        rset = {int(r) for r in row_ids}
        ridx = {int(r): i for i, r in enumerate(row_ids)}
        for c in range(g):
            a, b = int(indptr[c]), int(indptr[c + 1])
            if b == a:
                continue
            rows = X["indices"][a:b]
            m = np.fromiter((int(x) in rset for x in rows), bool, len(rows))
            if not m.any():
                continue
            sel = np.fromiter((ridx[int(x)] for x in rows[m]), np.int64, m.sum())
            out[sel, c] = X["data"][a:b][m]
    return out


def load_ntc_pool(h5ad_path, control, hvg_genes):
    """NTC pool raw counts -> norm_log1p(11940) in context space -> project to 8192 order."""
    with h5py.File(h5ad_path, "r") as f:
        ctx_genes = read_var_names(f["var"])
        tg = read_obs_str(f, "target_gene")
        ntc_ids = [i for i, t in enumerate(tg) if t == control]
        assert ntc_ids, f"{h5ad_path}: no '{control}' rows"
        X = read_csr_rows(f, ntc_ids)  # (n_ntc, ctx_g) raw counts
    sums = X.sum(1, keepdims=True)
    sums[sums == 0] = 1.0
    X = np.log1p(X / sums * NORM_TARGET_SUM)
    pos = {g: i for i, g in enumerate(ctx_genes)}
    col = np.full(len(hvg_genes), -1, np.int64)
    for j, g in enumerate(hvg_genes):
        i = pos.get(g)
        if i is not None:
            col[j] = i
    full = np.zeros((X.shape[0], len(hvg_genes)), np.float32)
    full[:, col >= 0] = X[:, col[col >= 0]]
    return full, ctx_genes


def read_targets_submit():
    with open(ROOT / "data/raw/controls/pert_counts.csv") as fh:
        rows = list(csv.DictReader(fh))
    return [r["target_gene"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default=",".join(list(SUBMIT_CTX) + list(BENCH_CTX)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-windows", type=int, default=2, help="64-cell ctrl windows per target")
    ap.add_argument("--max-targets", type=int, default=0, help="0 = all targets (smoke uses 2)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--run-dir", default=str(RUN_DIR),
                    help="run dir holding var_dims.pkl/pert_onehot_map.pt (default: mother arm)")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint to load (default <run-dir>/checkpoints/best.ckpt)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.device, args.n_windows = "cpu", 1
        args.max_targets = 2
        args.outdir = args.outdir or str(ROOT / "tmp/v06_dl_smoke")
        args.contexts = ",".join(args.contexts.split(",")[:1])
    outdir = Path(args.outdir or OUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    ckpt = Path(args.ckpt) if args.ckpt else run_dir / "checkpoints/best.ckpt"
    t0 = time.time()

    # mapping table + live var order (bit-exactness asserted)
    mrows = list(csv.DictReader(open(MAPPING_CSV)))
    hvg_genes = [r["hvg_gene"] for r in mrows]
    with open(run_dir / "var_dims.pkl", "rb") as fh:
        vd = pickle.load(fh)
    vd_genes = [g.decode() if isinstance(g, bytes) else str(g) for g in vd["gene_names"]]
    assert vd_genes == hvg_genes, "mapping_run_v06.csv hvg_gene != live var_dims gene_names"
    hpos = {g: i for i, g in enumerate(hvg_genes)}
    panel_idx = np.array([int(r["panel_idx"]) for r in mrows])
    in_panel = panel_idx >= 0

    # model + pert onehot map
    model = StateTransitionPerturbationModel.load_from_checkpoint(str(ckpt))
    device = args.device
    model = model.to(device).eval()
    pert_map = torch.load(run_dir / "pert_onehot_map.pt", map_location="cpu", weights_only=False)
    ntc_onehot = pert_map["non-targeting"]
    pert_dim = int(vd["pert_dim"])

    manifest = {"artifact": str(outdir), "ckpt": str(ckpt), "run_dir": str(run_dir),
                "n_windows": args.n_windows, "device": device, "contexts": {}}
    for ctx in args.contexts.split(","):
        if ctx in SUBMIT_CTX:
            ctrl_path, v03_path = SUBMIT_CTX[ctx]
            control, targets = "non-targeting", read_targets_submit()
        elif ctx in BENCH_CTX:
            ctrl_path, v03_path = BENCH_CTX[ctx]
            from src.models.bizard.swarm import load_cfg
            control = load_cfg()["scenarios"][ctx]["control"]
            targets = [str(t) for t in pd.read_json(BENCH / ctx / "targets.json")[0]]
        else:
            raise SystemExit(f"unknown context {ctx}")

        pool, ctx_genes = load_ntc_pool(ctrl_path, control, hvg_genes)
        basal = pool.mean(0)  # (8192,) ntc_mean caliber, 8192 log1p space
        print(f"[{ctx}] ntc_pool={len(pool)} control={control} targets={len(targets)}", flush=True)

        # v03 reference grid + row-order hard assert
        v03 = pd.read_parquet(v03_path, columns=["target_gene", "gene"])
        glist = v03["gene"].to_numpy()
        tlist_arr = v03["target_gene"].to_numpy()
        T, G = len(targets), len(glist) // len(set(glist[:4096]))
        # derive G from the first target block, then verify the whole tiling
        g0 = v03["gene"].to_numpy()[: len(np.unique(v03["gene"].to_numpy()))]
        G = len(g0)
        assert len(v03) == len(targets) * G, f"grid {len(v03)} != {len(targets)}x{G}"
        assert (tlist_arr.reshape(-1, G) == np.array(targets)[:, None]).all(), "target order mismatch"
        assert (glist.reshape(-1, G) == g0[None, :]).all(), "gene grid not per-target identical"
        if args.max_targets:  # smoke: keep the first target blocks of the same grid
            v03 = v03.iloc[: args.max_targets * G]
            targets = targets[: args.max_targets]

        # grid genes -> 8192 columns (out-of-set -> 0.0)
        gcols = np.array([hpos.get(g, -1) for g in g0], np.int64)

        rng = np.random.default_rng(SEED + zlib.crc32(ctx.encode()))
        n_fb = 0
        delta_rows = np.empty((len(targets), G), np.float32)
        for ti, tgt in enumerate(targets):
            oh = pert_map.get(str(tgt))
            if oh is None:
                oh, n_fb = ntc_onehot, n_fb + 1  # official fallback: control pert onehot
            acc = np.zeros((1, len(hvg_genes)), np.float32)
            for _ in range(args.n_windows):
                idx = rng.choice(len(pool), size=CELL_SET_LEN, replace=True)
                batch = {
                    "ctrl_cell_emb": torch.from_numpy(pool[np.sort(idx)]).float().to(device),
                    "pert_emb": torch.from_numpy(
                        np.repeat(oh.numpy() if isinstance(oh, torch.Tensor) else np.asarray(oh),
                                  CELL_SET_LEN, 0)).float().to(device),
                    "pert_name": [str(tgt)] * CELL_SET_LEN,
                    "batch": torch.zeros(CELL_SET_LEN, dtype=torch.long, device=device),
                }
                with torch.no_grad():
                    out = model.predict_step(batch, batch_idx=0, padded=False)
                preds = out["preds"].detach().cpu().numpy().astype(np.float32)
                np.clip(preds, CLIP_LO, CLIP_HI, out=preds)  # official clip on absolute preds
                acc += preds.sum(0)
            mean_pred = acc[0] / (args.n_windows * CELL_SET_LEN)
            delta8192 = (mean_pred - basal) / LN2  # log1p-delta -> log2FC
            proj = np.zeros(G, np.float32)
            ok = gcols >= 0
            proj[ok] = delta8192[gcols[ok]]
            delta_rows[ti] = proj
        n_fb_total = n_fb

        # optional mapping-table cross-check (submit panel order only; warn-only)
        if ctx.startswith("submit"):
            bad = [(mrows[i]["hvg_gene"], ctx_genes[panel_idx[i]])
                   for i in range(len(mrows)) if in_panel[i]
                   and ctx_genes[panel_idx[i]] != mrows[i]["hvg_gene"]]
            if bad:
                print(f"[{ctx}] WARN mapping panel_idx mismatch: {len(bad)} rows", flush=True)

        out = pd.DataFrame({
            "target_gene": np.repeat(np.array(targets, dtype=object), G),
            "gene": np.tile(g0.astype(object), len(targets)),
            "delta": delta_rows.reshape(-1),
        })
        assert len(out) == len(v03)
        opath = outdir / f"pred_dl_v06_{ctx}.parquet"
        out.to_parquet(opath, index=False)
        manifest["contexts"][ctx] = {
            "parquet": str(opath), "rows": int(len(out)), "targets": len(targets),
            "genes": G, "pert_fallback_control_onehot": int(n_fb_total),
            "v03_reference": str(v03_path), "delta_units": "log2FC (ln-delta / ln2)",
        }
        print(f"[{ctx}] wrote {opath} rows={len(out)} fallback={n_fb_total} "
              f"({time.time()-t0:.0f}s)", flush=True)

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] {outdir}/manifest.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
