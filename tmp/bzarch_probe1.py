# Probe 1: vcc2025 Flex_1/2/3 相似度 — 跨组 NTC 相似度 + 扰动响应相关
# 如果 Flex 组间 NTC 几乎相同 → context 差异小, 学不到 context-迁移; 
# 如果响应跨组相关高 → 均值基线很强, 学习空间=残差
import h5py, numpy as np, pandas as pd, scipy.sparse as sp
f = h5py.File("data/public/vcc2025/adata_Training.h5ad", "r")
obs = f["obs"]
bcat = [c.decode() if isinstance(c, bytes) else str(c) for c in obs["batch"]["categories"][:]]
bcodes = obs["batch"]["codes"][:]
tcat = [c.decode() if isinstance(c, bytes) else str(c) for c in obs["target_gene"]["categories"][:]]
tcodes = obs["target_gene"]["codes"][:]

def group_pseudobulk(grp, target=None, max_cells=20000):
    m = np.zeros(len(bcodes), dtype=bool)
    for i, c in enumerate(bcodes):
        if bcat[c].startswith(grp) and (target is None or tcat[c] == target):
            m[i] = True
    idx = np.where(m)[0]
    if len(idx) > max_cells:
        idx = np.random.default_rng(0).choice(idx, max_cells, replace=False)
    # stream rows
    rows = []
    indptr = f["X"]["indptr"][:]
    data = f["X"]["data"][:]
    indices = f["X"]["indices"][:]
    sub = sp.csr_matrix((data[indptr[idx]:indptr[idx+1]].sum() and np.concatenate(
        [data[indptr[i]:indptr[i+1]] for i in idx[:0]]), (0, 18080))
    )  # placeholder
    return idx

# simpler: use groupby sums via indptr slicing
indptr = f["X"]["indptr"][:]
data = f["X"]["data"][:]
indices = f["X"]["indices"][:]

def rows_sum(idx, out_len=18080):
    # sum sparse rows in idx — vectorized via np.add.reduceat on segments
    seg = np.concatenate([np.arange(indptr[i], indptr[i+1]) for i in idx])
    cols = indices[seg]
    vals = data[seg]
    out = np.zeros(out_len)
    np.add.at(out, cols, vals)
    return out

rng = np.random.default_rng(0)
def sample_idx(m, k=8000):
    idx = np.where(m)[0]
    return rng.choice(idx, min(k, len(idx)), replace=False) if len(idx) > k else idx

ntc_code = tcat.index("non-targeting")
groups = {}
for grp in ("Flex_1", "Flex_2", "Flex_3"):
    m = np.array([bcat[c].startswith(grp) for c in bcodes]) & (tcodes == ntc_code)
    idx = sample_idx(m, 8000)
    groups[grp] = rows_sum(idx) / len(idx)
    print(grp, "ntc cells sampled:", len(idx))

g1, g2, g3 = groups["Flex_1"], groups["Flex_2"], groups["Flex_3"]
def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
print("NTC cos Flex1-2:", cos(g1, g2), "1-3:", cos(g1, g3), "2-3:", cos(g2, g3))
print("NTC mean-profile correlation:", np.corrcoef(g1, g2)[0,1], np.corrcoef(g1, g3)[0,1])
np.save("tmp/bzarch_ntc_means.npy", np.stack([g1, g2, g3]))
f.close()
