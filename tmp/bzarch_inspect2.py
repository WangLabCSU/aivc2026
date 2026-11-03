import h5py, numpy as np, collections
f = h5py.File("data/public/vcc2025/adata_Training.h5ad", "r")
X = f["X"]
print("X keys:", list(X.keys()))
# anndata CSR stores shape in attrs
print("X attrs:", dict(X.attrs))
for k in ("shape",):
    if k in X:
        print("shape:", X[k][:])
indptr = X["indptr"]
print("n_obs = indptr_len-1:", indptr.shape)
print("n_var from indices max? use var")
print("var rows:", f["var"]["_index"].shape)
obs = f["obs"]
print("obs keys:", list(obs.keys()))
for k in obs.keys():
    if "categories" in obs[k]:
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in obs[k]["categories"][:]]
        print(f"  {k}: n_cat={len(cats)}", cats[:8])
        if k in ("context",):
            codes = obs[k]["codes"][:]
            cnt = collections.Counter(cats[c] for c in codes)
            print("   counts:", dict(cnt))
codes = obs["batch"]["codes"][:]
cats = [c.decode() if isinstance(c, bytes) else str(c) for c in obs["batch"]["categories"][:]]
cnt = collections.Counter(cats[c] for c in codes)
print("batch: n=", len(cnt), "first5:", dict(list(cnt.items())[:5]))
print("encoding-type:", X.attrs.get("encoding-type"), "shape attr:", X.attrs.get("shape"))
