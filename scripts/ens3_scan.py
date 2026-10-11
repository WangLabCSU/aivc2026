"""三/四源 Δ̂ 级 ensemble 扫描（Q_HepG2）：heur(dualsrc) × fusion v3.5 × CCRN-loco × Bzn-skip。

语义沿用 ens_alpha_scan.py：加权 skipna 均值（union，单源独有对保留）；
新增 sd 匹配——每源先除以自身 finite 对内 sd 再乘 heur 基准 sd（bzn 幅度 0.008 远小于
其余 0.03，不匹配会被 skipna 均值静默稀释）。λ 语义不变：pred ⊕ 2^(λ·Δ̂)。

用法:
  python scripts/ens3_scan.py run --w 0.4,0.4,0.2,0 --lam 4.5 --tag ens3_a
  python scripts/ens3_scan.py eval --tag ens3_a
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
ENV_BIN = Path("/data/home/wsx/miniconda3/envs/aivc2026/bin")
OUT = ROOT / "outputs/metrics/local_bench"
SCAN = ROOT / "data/interim/l15_fusion/v35/ensemble_scan"
REF = ROOT / "data/interim/local_bench/Q_HepG2/ref.h5ad"
BENCH = ROOT / "data/interim/local_bench/Q_HepG2"
LAM = 4.5
MAX_FOLD = 8.0
N_CELLS = 400
SEED = 0
CTRL = "non-targeting"
CFG = ["--preset", "vcc2026", "--pert-col", "target_gene", "--control", CTRL,
       "--input-type", "counts", "--set", "de.backend=pdex"]

_PANEL = None


def panel_of() -> pd.Index:
    global _PANEL
    if _PANEL is None:
        _r = ad.read_h5ad(REF, backed="r")
        _PANEL = pd.Index(_r.var_names.astype(str))
        _r.file.close()
    return _PANEL


def load_pivots() -> dict[str, pd.DataFrame]:
    targets = json.loads((BENCH / "targets.json").read_text())
    srcs = {
        "heur": pd.read_parquet(SCAN / "delta_heur.parquet"),
        "fusion": pd.read_parquet(SCAN / "delta_fusion.parquet"),
        "ccrn": pd.read_parquet(ROOT / "data/interim/bizard/ccrn/pred_ccrn1_loco_bench_hepg2.parquet"),
        "bzn": pd.read_parquet(ROOT / "data/interim/bizard/bizardnet/pred_skip_bench_hepg2.parquet"),
    }
    panel = panel_of()
    out = {}
    for k, df in srcs.items():
        p = df[df.target_gene.isin(targets)].pivot(
            index="target_gene", columns="gene", values="delta").reindex(columns=panel)
        out[k] = p
    return out


def sd_matched_blend(srcs: dict[str, pd.DataFrame], w: np.ndarray,
                     tags: list[str]) -> pd.DataFrame:
    """sd 匹配到 heur 后的加权 skipna 均值，返回长表。"""
    piv = [srcs[t] for t in tags]
    idx = piv[0].index
    for p in piv[1:]:
        idx = idx.union(p.index)
    cols = piv[0].columns
    for p in piv[1:]:
        cols = cols.union(p.columns)
    mats = [p.reindex(index=idx, columns=cols).to_numpy() for p in piv]
    sd0 = np.nanstd(mats[0])
    num = np.zeros_like(mats[0])
    den = np.zeros_like(mats[0])
    for w_i, m in zip(w, mats):
        if w_i <= 0:
            continue
        s = sd0 / np.nanstd(m)
        num += w_i * s * np.nan_to_num(m, nan=0.0)
        den += w_i * np.isfinite(m)
    ens = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    long = pd.DataFrame({"target_gene": np.repeat(idx.to_numpy(), len(cols)),
                         "gene": np.tile(cols.to_numpy(), len(idx)),
                         "delta": ens.astype(np.float32).ravel()})
    return long[np.isfinite(long["delta"])].reset_index(drop=True)


def build_pred(delta: pd.DataFrame, outdir: Path, lam: float) -> Path:
    a = ad.read_h5ad(REF)
    panel = pd.Index(a.var_names.astype(str))
    targets = json.loads((BENCH / "targets.json").read_text())
    mask = a.obs["target_gene"].astype(str).to_numpy() == CTRL
    Xntc = sp.csr_matrix(a[mask].X)
    del a
    rng = np.random.default_rng(SEED)
    dmap = delta.set_index(["target_gene", "gene"])["delta"]
    pred_perts = [CTRL] + targets
    blocks = []
    for p in pred_perts:
        M = Xntc[rng.choice(Xntc.shape[0], N_CELLS, replace=True)].copy()
        if p in dmap.index.get_level_values(0):
            dd = dmap.loc[p]
            if isinstance(dd.index, pd.MultiIndex):
                dd = dd.xs(p, level=0)
            dv = dd.reindex(panel).to_numpy()
            f = np.power(2.0, lam * np.nan_to_num(dv, nan=0.0)).astype(np.float32)
            orig = M.data.copy()
            M.data = np.minimum(M.data * f[M.indices], MAX_FOLD * orig)
        M.data = np.clip(np.rint(M.data), 0, None).astype(np.float32)
        M.eliminate_zeros()
        blocks.append(M)
    X = sp.vstack(blocks, format="csr").astype(np.float32)
    X.eliminate_zeros()
    pred = ad.AnnData(X=X, obs=pd.DataFrame(
        {"target_gene": np.repeat(pred_perts, N_CELLS), "context": "hepg2"},
        index=[f"p{i}_c{j}" for i in range(len(pred_perts)) for j in range(N_CELLS)]))
    pred.var_names = panel.to_numpy()
    pred.var.index.name = None
    outdir.mkdir(parents=True, exist_ok=True)
    pp = outdir / "Q_HepG2_pred.h5ad"
    pred.write_h5ad(pp, compression="gzip")
    return pp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "eval"])
    ap.add_argument("--w", type=str, default="0.5,0.5,0,0")
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--tag", type=str, required=True)
    args = ap.parse_args()

    tag = args.tag
    outdir = OUT / tag
    if args.mode == "run":
        t0 = time.time()
        w = np.array([float(x) for x in args.w.split(",")])
        srcs = load_pivots()
        tags = ["heur", "fusion", "ccrn", "bzn"]
        long = sd_matched_blend(srcs, w, tags)
        long.to_parquet(SCAN / f"delta_{tag}.parquet")
        pp = build_pred(long, outdir, args.lam)
        meta = {"w": args.w, "lam": args.lam, "rows": int(len(long)),
                "absmean": float(long.delta.abs().mean()),
                "src_sd": {t: float(np.nanstd(srcs[t].to_numpy())) for t in tags}}
        (SCAN / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2))
        print(f"[{tag}] rows={len(long)} absmean={meta['absmean']:.4f} "
              f"({time.time()-t0:.0f}s) -> {pp}", flush=True)
    else:
        cmd = [str(ENV_BIN / "cell-eval2"), "run", "-ap",
               str(outdir / "Q_HepG2_pred.h5ad"), "-ar", str(REF),
               *CFG, "-o", str(outdir / "Q_HepG2_eval")]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        print(f"[{tag}] eval done", flush=True)


if __name__ == "__main__":
    main()
