import json, h5py
f = h5py.File("data/public/vcc2025/adata_Training.h5ad", "r")
out = {}
out["n_obs"] = f["X"]["data"].shape[0] if "X" in f and "data" in f["X"] else None
obs = f["obs"]
out["obs_cols"] = list(obs.keys())
for k in obs.keys():
    sub = obs[k]
    if "categories" in sub:
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in sub["categories"][:]]
        out[f"obs_{k}"] = {"n_cat": len(cats), "first10": cats[:10]}
    else:
        out[f"obs_{k}"] = "numeric/var"
for k in ("var", "uns", "obsm", "layers", "raw"):
    if k in f:
        try:
            out[f"{k}_keys"] = list(f[k].keys())[:20]
        except Exception:
            pass
X = f["X"]
out["X_keys"] = list(X.keys()) if hasattr(X, "keys") else str(type(X))
if "shape" in X:
    out["X_shape"] = list(X["shape"])
# context-like column detail
print(json.dumps(out, indent=1, default=str)[:4000])
