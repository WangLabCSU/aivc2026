#!/usr/bin/env python
"""调解探针矛盾: 同一数据 undeduped pandas merge vs deduped intersect1d, corr / |abs| corr / spearman."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

SIG = "/data/resource/wanglab/aivc/aivc2026/data/interim/signatures"
cols = ["target_gene", "gene", "logfc"]
g = pd.read_parquet(f"{SIG}/gwps.parquet", columns=cols)
h = pd.read_parquet(f"{SIG}/hepg2_all.parquet", columns=cols)
print("gwps", g.shape, "dupes:", g.duplicated(["target_gene","gene"]).sum())
print("hepg2", h.shape, "dupes:", h.duplicated(["target_gene","gene"]).sum())

m = g.merge(h, on=["target_gene","gene"], suffixes=("_g","_h"))
print("merged (undeduped):", len(m))
print("corr(logfc_g, logfc_h):", np.corrcoef(m.logfc_g, m.logfc_h)[0,1])
print("corr(|logfc_g|, |logfc_h|):", np.corrcoef(np.abs(m.logfc_g), np.abs(m.logfc_h))[0,1])
sp = spearmanr(m.logfc_g, m.logfc_h).statistic
print("spearman global:", sp)

# 去重后 (mean) 再算
gm = g.groupby(["target_gene","gene"], sort=False)["logfc"].mean().reset_index()
hm = h.groupby(["target_gene","gene"], sort=False)["logfc"].mean().reset_index()
m2 = gm.merge(hm, on=["target_gene","gene"])
print("\ndeduped pairs:", len(m2))
print("corr:", np.corrcoef(m2.logfc_x, m2.logfc_y)[0,1])
print("corr(abs):", np.corrcoef(np.abs(m2.logfc_x), np.abs(m2.logfc_y))[0,1])
print("spearman global:", spearmanr(m2.logfc_x, m2.logfc_y).statistic)
print("mae direct:", np.mean(np.abs(m2.logfc_x - m2.logfc_y)))
print("mae zero:", np.mean(np.abs(m2.logfc_y)))
# 分位数检查 y 分布
print("\ny |logfc| mean:", np.abs(m2.logfc_y).mean(), "std:", m2.logfc_y.std())
print("y logfc quantiles:", np.quantile(m2.logfc_y, [0.01,0.25,0.5,0.75,0.99]))
print("x logfc quantiles:", np.quantile(m2.logfc_x, [0.01,0.25,0.5,0.75,0.99]))
