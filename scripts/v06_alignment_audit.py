#!/usr/bin/env python
"""Read-only v0.6 encoder-init alignment audit.

Quantifies, for each of the 8 ST-HVG-Replogle best.ckpt candidates:
  - gene-space overlap (var_dims.gene_names vs each of 6 context panels + competition 18533 panel)
  - pert-space coverage (pert_names / pert_onehot_map vs the 300 competition targets)

Strictly read-only: opens model files read-only, writes nothing under data/public/models/.
Output: outputs/v06_alignment_audit/alignment_audit.json + alignment_table.csv
"""
import glob
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
OUT = ROOT / "outputs/v06_alignment_audit"
OUT.mkdir(parents=True, exist_ok=True)

PANELS = {
    "gwps": ROOT / "data/public/scperturb/Replogle2022_K562_gwps.h5ad",
    "k562_ess": ROOT / "data/public/scperturb/Replogle2022_K562_essential.h5ad",
    "rpe1": ROOT / "data/public/scperturb/Replogle2022_rpe1.h5ad",
    "jurkat": ROOT / "data/public/scperturb/Nadig2024_jurkat.h5ad",
    "hepg2": ROOT / "data/public/scperturb/Nadig2024_hepg2.h5ad",
    "hct116": ROOT / "data/public/xatlas/HCT116_pooled.h5ad",
}


def read_var_names(v):
    """Read gene names from an h5ad var group, handling sparse-string encoding."""
    for key in ("gene_name", "_index"):
        if key in v:
            o = v[key]
            if isinstance(o, h5py.Dataset):
                arr = o[:]
            elif isinstance(o, h5py.Group):
                # categorical / categorical-encoding storage: try codes+categories
                if "categories" in o:
                    arr = o["categories"][:]
                else:
                    continue
            else:
                continue
            out = []
            for x in arr:
                if isinstance(x, bytes):
                    out.append(x.decode())
                elif isinstance(x, np.ndarray) and x.dtype.kind in "SU":
                    # h5py 3.x variable-length strings can come back as np.bytes_ arrays
                    out.append(x[0].decode() if x.dtype.kind == "S" else str(x[0]))
                else:
                    out.append(str(x))
            return out
    return None


def main():
    # ---- panel gene sets ----
    panel_genes = {}
    for name, p in PANELS.items():
        with h5py.File(p, "r") as f:
            names = read_var_names(f["var"])
        assert names is not None, f"cannot read var names for {name}"
        panel_genes[name] = set(names)
        print(f"[panel] {name}: {len(panel_genes[name])} unique genes", flush=True)

    # ---- competition 18533-gene evaluation panel ----
    # The local bench reference/eval space: use Validation h5ad var index (18080)
    # AND the delta-parquet gene column (the actual grid the swarm fuses on).
    with h5py.File(ROOT / "data/public/vcc2025/adata_Validation.h5ad", "r") as f:
        v = f["var"]
        comp_genes = set(
            x.decode() if isinstance(x, bytes) else str(x) for x in v["_index"][:]
        )
    print(f"[comp] adata_Validation var: {len(comp_genes)} genes", flush=True)

    dfa = pd.read_parquet(
        ROOT / "data/interim/bizard/v03/agg_delta_submit_A.parquet", columns=["gene"]
    )
    grid_genes = set(dfa["gene"].unique())
    print(f"[comp] submit_A delta gene grid: {len(grid_genes)} genes", flush=True)

    # ---- 300 competition targets ----
    dft = pd.read_parquet(
        ROOT / "data/interim/bizard/v03/agg_delta_submit_A.parquet",
        columns=["target_gene"],
    )
    targets = set(dft["target_gene"].unique())
    print(f"[comp] competition targets: {len(targets)}", flush=True)

    # ---- per-ckpt audit ----
    rows = []
    detail = {}
    ckpt_dirs = sorted(glob.glob(str(ROOT / "data/public/models/ST-HVG-Replogle/*/*")))
    for d in ckpt_dirs:
        if not (Path(d) / "var_dims.pkl").exists():
            continue
        tag = str(Path(d).relative_to(ROOT / "data/public/models/ST-HVG-Replogle"))
        with open(Path(d) / "var_dims.pkl", "rb") as fh:
            vd = pickle.load(fh)
        genes = set(str(x) for x in vd["gene_names"])
        perts = set(str(x) for x in vd["pert_names"])
        row = {
            "ckpt": tag,
            "n_genes": len(genes),
            "n_perts": len(perts),
        }
        # gene overlap vs each panel
        for name, gset in panel_genes.items():
            inter = len(genes & gset)
            row[f"ovl_{name}"] = round(inter / len(genes), 4)
            row[f"inter_{name}"] = inter
            row[f"panel_{name}_n"] = len(gset)
        # overlap vs competition eval spaces
        row["ovl_validation18080"] = round(len(genes & comp_genes) / len(genes), 4)
        row["inter_validation18080"] = len(genes & comp_genes)
        row["ovl_submitgrid"] = round(len(genes & grid_genes) / len(genes), 4)
        row["inter_submitgrid"] = len(genes & grid_genes)
        # pert coverage of the 300 competition targets
        row["target_cover"] = len(targets & perts)
        row["target_cover_frac"] = round(len(targets & perts) / len(targets), 4)
        rows.append(row)
        print(f"[ckpt] {tag}: genes={len(genes)} perts={len(perts)} "
              f"target_cover={row['target_cover']}/300 "
              f"ovl_gwps={row['ovl_gwps']} ovl_submitgrid={row['ovl_submitgrid']}",
              flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "alignment_table.csv", index=False)

    # ---- conclusion heuristics ----
    # risk flag: any panel overlap below 0.5 is a silent-mismatch risk for encoder-init
    ovl_cols = [c for c in table.columns if c.startswith("ovl_")]
    table["min_panel_ovl"] = table[ovl_cols].min(axis=1)
    table["risk_flag"] = np.where(table["min_panel_ovl"] < 0.5, "HIGH", "low")
    best = table.sort_values(
        ["target_cover_frac", "min_panel_ovl"], ascending=False
    ).iloc[0]
    summary = {
        "date": "2026-10-15",
        "n_ckpts_audited": len(rows),
        "panels": {k: len(v) for k, v in panel_genes.items()},
        "competition_validation_genes": len(comp_genes),
        "submit_grid_genes": len(grid_genes),
        "n_targets": len(targets),
        "best_ckpt_by_cover_then_overlap": str(best["ckpt"]),
        "best_target_cover": int(best["target_cover"]),
        "best_min_panel_ovl": float(best["min_panel_ovl"]),
        "high_risk_ckpts": table.loc[table["risk_flag"] == "HIGH", "ckpt"].tolist(),
    }
    payload = {
        "summary": summary,
        "table": rows,
        "panel_gene_counts": {k: len(v) for k, v in panel_genes.items()},
        "competition_validation_genes_n": len(comp_genes),
        "submit_grid_genes_n": len(grid_genes),
        "n_targets": len(targets),
    }
    (OUT / "alignment_audit.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
