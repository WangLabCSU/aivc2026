"""Build cell-eval2 pred.h5ad from State zeroshot hepg2 raw_infer output.

- 模拟输出 obsm['X_hvg']: log1p(CPM1e4) 空间, 2,000 HVG, obs 顺序=ref。
- counts 还原: expm1 / 1e4 * 源细胞 lib size, 四舍五入为整数 counts。
- 非 HVG 基因(7,624): 用 ref NTC 均值填充(模型不预测)。
- NTC 预测块: ref NTC 均值行 tile x400(与我们 lambda pipeline 的 meanrows 约定一致)。
"""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

ROOT = "/data/resource/wanglab/aivc/aivc2026"
out = ad.read_h5ad(f"{ROOT}/outputs/metrics/local_bench/state_zh_hepg2/raw_infer.h5ad")
ref = ad.read_h5ad(f"{ROOT}/data/interim/local_bench/Q_HepG2/ref.h5ad")
lib = np.load(f"{ROOT}/tmp/state_zh_libsizes.npy")

S = out.obsm["X_hvg"]
counts = np.clip(np.rint(np.expm1(np.clip(S, 0, None)) / 1e4 * lib[:, None]), 0, None).astype(np.float32)

gn = list(out.var_names)
gidx = {g: i for i, g in enumerate(gn)}
panel = np.array(ref.var_names.astype(str))
Xr = ref.X.toarray() if sp.issparse(ref.X) else np.asarray(ref.X)
ctrl = "non-targeting"
mean = np.rint(Xr[(ref.obs["target_gene"] == ctrl).to_numpy()].mean(0)).astype(np.float32)
targets = [t for t in ref.obs["target_gene"].astype(str).unique() if t != ctrl]
tg = out.obs["target_gene"].astype(str).to_numpy()

is_hvg = np.array([g in gidx for g in panel])
hvg_cols = np.array([gidx[g] for g in panel[is_hvg]])
n = 400

blocks, names, idx = [], [], []
bi = 0


def add(M, name):
    global bi
    blocks.append(sp.csr_matrix(M))
    names.append(name)
    idx.extend(f"p{bi}_c{j}" for j in range(M.shape[0]))
    bi += 1


add(np.tile(mean, (n, 1)), ctrl)  # NTC block = 均值行
for p in targets:
    m = tg == p
    M = np.zeros((int(m.sum()), len(panel)), dtype=np.float32)
    M[:, ~is_hvg] = mean[~is_hvg]
    M[:, is_hvg] = counts[m][:, hvg_cols]
    add(M, p)

sizes = [b.shape[0] for b in blocks]
X = sp.vstack(blocks, format="csr").astype(np.float32)
pred = ad.AnnData(X=X, obs=pd.DataFrame(
    {"target_gene": np.repeat(names, sizes), "context": "Q_HepG2"}, index=idx))
pred.var_names = panel
pred.var.index.name = None
pp = f"{ROOT}/outputs/metrics/local_bench/state_zh_hepg2/Q_HepG2_pred.h5ad"
pred.write_h5ad(pp, compression="gzip")
print("pred:", pred.shape, "perts:", len(names), "hvg mean", round(float(X[:, is_hvg].mean()), 3),
      "nonhvg mean", round(float(X[:, ~is_hvg].mean()), 3))
print("written", pp)
