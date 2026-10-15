"""Six-context probe: how does submit_A delta map onto the 5 bench contexts + 2 new?"""
import json

import pandas as pd

ROOT = "/data/resource/wanglab/aivc/aivc2026"
BASE = ROOT + "/data/interim/bizard/v03/gen2/"

d = pd.read_parquet(BASE + "pred_g2_c3_submit_A.parquet",
                    columns=["target_gene", "gene", "delta"])
print("submit_A: targets", d.target_gene.nunique(), "genes", d.gene.nunique(),
      "rows", len(d))
tgA = set(d.target_gene.unique())

for ctx in ["P_H1", "Q_HepG2", "R_Jurkat", "S_RPE1", "S_HCT116"]:
    t = json.loads(open(f"{ROOT}/data/interim/local_bench/{ctx}/targets.json").read())
    dc = pd.read_parquet(BASE + f"pred_g2_c3_{ctx}.parquet", columns=["target_gene"])
    print(ctx, "bench_targets", len(t), "in_A", len(set(t) & tgA),
          "| ctx_delta_targets", dc.target_gene.nunique())

print("gene grid of submit_A:", d.gene.nunique())
print("gene grid of P_H1:",
      pd.read_parquet(BASE + "pred_g2_c3_P_H1.parquet", columns=["gene"]).gene.nunique())
