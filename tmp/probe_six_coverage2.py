"""Probe 2: submit_A delta vs panel bench-target sets; agg-delta union vs k562_ess."""
import glob
import json

import h5py
import pandas as pd

ROOT = "/data/resource/wanglab/aivc/aivc2026"
G = ROOT + "/data/interim/bizard/v03/gen2/"
V = ROOT + "/data/interim/bizard/v03/"

sub = pd.read_parquet(G + "pred_g2_c3_submit_A.parquet", columns=["target_gene"])
tg = set(sub.target_gene.unique())
print("submit targets:", len(tg))

for ctx in ["P_H1", "Q_HepG2", "R_Jurkat", "S_RPE1", "S_HCT116"]:
    t = set(json.load(open(f"{ROOT}/data/interim/local_bench/{ctx}/targets.json")))
    print(f"{ctx}: bench{len(t)} ∩ submit300 = {len(t & tg)}")

with h5py.File(ROOT + "/data/public/scperturb/Replogle2022_K562_essential.h5ad") as f:
    g = f["obs/gene"]
    cats = [c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]]
    vc = pd.Series(pd.Categorical.from_codes(g["codes"][:], cats)).astype(str).value_counts()
ess = set(vc[vc >= 50].index) - {"non-targeting"}
print("ess eligible:", len(ess))

u = set()
for p in sorted(glob.glob(V + "agg_delta_*.parquet")):
    if "submit" in p:
        continue
    u |= set(pd.read_parquet(p, columns=["target_gene"]).target_gene.unique())
print("agg_delta union:", len(u), "∩ess:", len(u & ess))

with h5py.File(ROOT + "/data/public/scperturb/Replogle2022_K562_gwps.h5ad") as f:
    g = f["obs/gene"]
    cats = [c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]]
    vc2 = pd.Series(pd.Categorical.from_codes(g["codes"][:], cats)).astype(str).value_counts()
gw = set(vc2[vc2 >= 50].index) - {"non-targeting"}
print("gwps eligible:", len(gw), "| union∩gw:", len(u & gw), "| submit300∩gw:", len(tg & gw))

for g_ in ["ACLY", "ABCD1", "RPL3", "AAMP", "MYC"]:
    print("spot", g_, "| ess:", g_ in ess, "| gw:", g_ in gw, "| submit:", g_ in tg)
