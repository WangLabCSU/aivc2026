"""Probe 3: essential coverage of all pred_* families in v03 root + gen2.

Terminal selection A composition for the frozen baseline used agg_delta_{P_H1,Q_HepG2,R_Jurkat,S_RPE1}.parquet
plus gen2/pred_g2_c3_S_HCT116.parquet. Any replacement source for k562_ess must come from
v0.5 lineage files; here we enumerate every pred_*.parquet family's target coverage vs
the 1832 essential-eligible perts.
"""
import glob
import os
import re

import h5py
import pandas as pd

ROOT = "/data/resource/wanglab/aivc/aivc2026"
V = ROOT + "/data/interim/bizard/v03/"
G = V + "gen2/"

with h5py.File(ROOT + "/data/public/scperturb/Replogle2022_K562_essential.h5ad") as f:
    g = f["obs/gene"]
    cats = [c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]]
    vc = pd.Series(pd.Categorical.from_codes(g["codes"][:], cats)).astype(str).value_counts()
ess = set(vc[vc >= 50].index) - {"non-targeting"}

fam = {}
for p in sorted(glob.glob(V + "pred_*.parquet")) + sorted(glob.glob(G + "pred_*.parquet")):
    name = os.path.basename(p)
    m = re.match(r"pred_(.+?)_(P_H1|Q_HepG2|R_Jurkat|S_RPE1|S_HCT116|submit_[ABC])\.parquet", name)
    key = m.group(1) if m else name
    fam.setdefault(key, []).append((name, p))

for key in sorted(fam):
    files = fam[key]
    # union of targets across the 5 per-context files of this family (skip submit)
    u = set()
    for name, p in files:
        if "submit" in name:
            continue
        u |= set(pd.read_parquet(p, columns=["target_gene"]).target_gene.unique())
    n_ctx = sum(1 for name, _ in files if "submit" not in name)
    print(f"{key:24s} ctx_files={n_ctx} union_targets={len(u):4d} ∩ess={len(u & ess):4d}")
