# iter023 诊断: 为何 bizard 头 vs truth 零配对 / topk 零配对
import pandas as pd, numpy as np

bz = pd.read_parquet("data/interim/bizard/v02/agg_delta_submit_A.parquet")
truth = pd.read_parquet("tmp/q_hepg2_truth_lfc_top50.parquet")
perts = pd.read_csv("data/raw/controls/pert_counts.csv")["target_gene"].astype(str).tolist()

print("bz shape:", bz.shape, "cols:", list(bz.columns))
print("bz dtypes:", {c: str(t) for c, t in bz.dtypes.items()})
for c in bz.columns:
    if bz[c].dtype == object or str(bz[c].dtype) == "category":
        print(c, "sample:", list(bz[c].dropna().astype(str).unique()[:5]))

tt = set(truth.target.astype(str))
print("truth targets n=", len(tt), "sample:", list(tt)[:5])
print("perts n=", len(set(perts)), "sample:", perts[:5])
print("truth∩perts:", len(tt & set(perts)))

if "target_gene" in bz.columns:
    bt = set(bz.target_gene.dropna().astype(str))
    print("bz targets n=", len(bt), "bz∩truth:", len(bt & tt), "bz∩perts:", len(bt & set(perts)))

tg = set(truth.gene.astype(str))
if "gene" in bz.columns:
    bg = set(bz.gene.dropna().astype(str))
    print("bz genes n=", len(bg), "bz∩truth genes:", len(bg & tg))
