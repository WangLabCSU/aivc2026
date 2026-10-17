
import anndata as ad, numpy as np
from scipy import sparse
import sys
gamma = float(sys.argv[1])
out = f"/data/resource/wanglab/aivc/aivc2026/outputs/predictions/sharpen_g{gamma}/Q_HepG2_pred.h5ad"
import os; os.makedirs(os.path.dirname(out), exist_ok=True)
A = ad.read_h5ad("/data/resource/wanglab/aivc/aivc2026/outputs/metrics/local_bench/dualdeep_lam70mf14_Q/Q_HepG2_pred.h5ad")
X = A.X.tocsr(); obs = A.obs
nt = np.where((obs.target_gene=="non-targeting").values)[0]
mu_g = np.asarray(X[nt].mean(axis=0)).ravel()
new = X.copy()
for t in obs.target_gene.unique():
    if t == "non-targeting": continue
    idx = np.where((obs.target_gene==t).values)[0]
    if len(idx)==0: continue
    M = X[idx].toarray()
    mu = M.mean(axis=0)
    M2 = np.clip(mu + (M - mu)*gamma, 0, None)
    new[idx] = np.rint(M2)
A.X = sparse.csr_matrix(new)
A.write_h5ad(out)
print("wrote", out)
