import h5py, numpy as np, collections
# Check Test + Validation structure (contexts for the challenge)
for fn in ("adata_Test.h5ad", "adata_Validation.h5ad"):
    f = h5py.File(f"data/public/vcc2025/{fn}", "r")
    X = f["X"]
    print(fn, "shape:", X.attrs.get("shape"))
    obs = f["obs"]
    print(" obs keys:", list(obs.keys()))
    for k in obs.keys():
        if "categories" in obs[k]:
            cats = [c.decode() if isinstance(c, bytes) else str(c) for c in obs[k]["categories"][:]]
            print(f"  {k}: n_cat={len(cats)}", cats[:8])
    f.close()
# pert counts
import pandas as pd
for s in ("Training", "Validation", "Test"):
    pc = pd.read_csv(f"data/public/vcc2025/pert_counts_{s}.csv")
    print(s, "pert_counts:", pc.shape, list(pc.columns), pc["target_gene"].nunique() if "target_gene" in pc else "")
# gene names overlap
gn_official = pd.read_csv("data/public/vcc2025/gene_names.csv")
gn_ctrl = pd.read_csv("data/raw/controls/gene_names.csv")
print("official gene_names:", gn_official.shape, "controls gene_names:", gn_ctrl.shape)
a = set(gn_official.iloc[:,0].astype(str)); b = set(gn_ctrl.iloc[:,0].astype(str))
print("overlap:", len(a & b), "/", len(b))
