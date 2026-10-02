"""E2 计数级 ensemble：pred_ens = rint((pred_heur_lam25 + pred_fus_lam25)/2)（按 target_gene 对齐块）。

两臂：g13b_lam2.5_qhepg2（启发式 [gwps,norman] λ2.5, Q overall 0.2668）
     l15v2s1_fused_lam2.5（fusion v2 LOCO-HepG2 λ2.5, Q overall 0.3222）
对齐后全体 block（含 NTC）取均值再四舍五入；输出 local_bench tag 目录并跑 cell-eval2 + bundle 归一化。
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
ENV_BIN = ROOT.parent.parent and Path("/data/home/wsx/miniconda3/envs/aivc2026/bin")
OUT = ROOT / "outputs/metrics/local_bench"
P_HEUR = OUT / "g13b_lam2.5_qhepg2/Q_HepG2_pred.h5ad"
P_FUS = OUT / "l15v2s1_fused_lam2.5/Q_HepG2_pred.h5ad"
REF = ROOT / "data/interim/local_bench/Q_HepG2/ref.h5ad"
BUNDLE = OUT / "bundles/Q_HepG2"
TAG = "e2_ens_heurfus_lam25_rint"

a1 = ad.read_h5ad(P_HEUR)
a2 = ad.read_h5ad(P_FUS)
assert list(a1.var_names) == list(a2.var_names), "var mismatch"
t1 = a1.obs["target_gene"].astype(str).to_numpy()
t2 = a2.obs["target_gene"].astype(str).to_numpy()
perts = list(dict.fromkeys(t1))  # heur 块序
assert set(perts) == set(t2), "pert set mismatch"
print(f"[ens] {len(perts)} perts, aligning blocks", flush=True)

X1, X2 = sp.csr_matrix(a1.X), sp.csr_matrix(a2.X)
ord1 = np.argsort(t1, kind="stable")
ord2 = np.argsort(t2, kind="stable")
X1s, X2s = X1[ord1], X2[ord2]
ts = t1[ord1]
assert np.array_equal(ts, t2[ord2]), "block alignment failed"
# 计数级融合: np.rint((a+b)/2)（banker's，任务规格 round 语义；half-up 在稀疏小计数上
# 系统性进位 → 质量膨胀 +9%、mse/pds 失真，已弃）
Xs = X1s + X2s
Xs.data = np.rint(Xs.data / 2.0).astype(np.float32)
Xs.eliminate_zeros()
# 恢复 heur 原 pert 块序：ord1 = argsort(t1)（原->排序），其逆置换把排序行映回原行
inv_ord = np.argsort(ord1, kind="stable")
Xout = Xs[inv_ord].tocsr()
Xout.eliminate_zeros()

obs = pd.DataFrame({"target_gene": t1, "context": "Q_HepG2"},
                   index=[f"p{i}_c{j}" for i in range(len(perts)) for j in range(400)])
pred = ad.AnnData(X=Xout, obs=obs)
pred.var_names = pd.Index(a1.var_names.astype(str))
pred.var.index.name = None
outdir = OUT / TAG
outdir.mkdir(parents=True, exist_ok=True)
pp = outdir / "Q_HepG2_pred.h5ad"
pred.write_h5ad(pp, compression="gzip")
print(f"[ens] wrote {pp} nnz={Xout.nnz}", flush=True)
# nnz 对比两臂
print(f"[ens] nnz heur={X1.nnz} fus={X2.nnz} ens={Xout.nnz} "
      f"mean_counts heur={X1.sum()/X1.shape[0]:.1f} fus={X2.sum()/X2.shape[0]:.1f} ens={Xout.sum()/Xout.shape[0]:.1f}", flush=True)

# cell-eval2 run（命令与 local_bench.run_celleval 一致）
ev = outdir / "Q_HepG2_eval"
cmd = [str(ENV_BIN / "cell-eval2"), "run", "-ap", str(pp), "-ar", str(REF),
       "--preset", "vcc2026", "--pert-col", "target_gene", "--control", "non-targeting",
       "--input-type", "counts", "--set", "de.backend=pdex", "-o", str(ev)]
subprocess.run(cmd, check=True)
agg = pd.read_csv(ev / "agg_results.csv")
row = agg[agg["statistic"] == "mean"].iloc[0]
print("[ens] raw:", {m.split('.')[-1]: round(float(row[m]), 4) for m in row.index if m != "statistic"}, flush=True)
# bundle 归一化打分
sp_ = outdir / "Q_HepG2_scored.csv"
subprocess.run([str(ENV_BIN / "cell-eval2"), "score", "--user-agg", str(ev / "agg_results.csv"),
                "--real-bundle", str(BUNDLE), "-o", str(sp_)], check=True)
sc = pd.read_csv(sp_).set_index("metric")
print("[ens] from_replicate:", sc["from_replicate"].to_dict(), flush=True)
