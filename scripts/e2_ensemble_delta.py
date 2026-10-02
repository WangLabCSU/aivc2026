"""E2b Δ̂ 级 ensemble：Δ_ens = mean(Δ_heur(gwps[,norman]), Δ_fusion_v2s1) → λ2.5 重建 pred → bench。

计数级融合已证伪（arithmetic≥geometric Jensen gap：方向相反基因产生 |λ(Δh-Δf)|/2 级虚假 LFC）。
Δ̂ 级融合在签名空间取均值后再走指数映射，无该偏差。
- Δ_heur: load_delta(g13b config=[gwps,norman] λ2.5 clip4.0)，norman 对 Q 靶零贡献（已验证）
- Δ_fusion: data/interim/l15_fusion/v2/pred_s1bench_hepg2_v2_fused.parquet（100 Q靶 × 9,624 panel）
- 聚合语义同 load_delta：按 (target,gene) 对均值，单源独有对保留单源值
- pred 构建 = local_bench.py 逐字逻辑（NTC bootstrap 采样 seed0、400/pert、λ2.5、无 max_fold 帽）
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
ENV_BIN = Path("/data/home/wsx/miniconda3/envs/aivc2026/bin")
sys.path.insert(0, str(ROOT))
from src.models.l1_signature.make_predictions import load_delta  # noqa: E402

OUT = ROOT / "outputs/metrics/local_bench"
TAG = "e2_ens_delta_lam25"
LAM = 2.5
REF = ROOT / "data/interim/local_bench/Q_HepG2/ref.h5ad"
BUNDLE = OUT / "bundles/Q_HepG2"

# 1) Δ_heur（g13b 同参）
gcfg = {"datasets": ["gwps", "norman"], "lambda_real": LAM, "lambda_generic": 0.15,
        "max_abs_delta": 4.0, "max_fold": 8.0, "stress_alpha": 0.0, "top_k": None}
a = ad.read_h5ad(REF)
panel = pd.Index(a.var_names.astype(str))
control = "non-targeting"
targets = [t for t in a.obs["target_gene"].astype(str).unique() if t != control]
d_heur, rep = load_delta(gcfg, panel, targets)
print(f"[dEns] heur delta rows {len(d_heur)}, covered {rep['covered_targets']}", flush=True)

# 2) Δ_fusion（长表 → 同聚合语义）
d_fus = pd.read_parquet(ROOT / "data/interim/l15_fusion/v2/pred_s1bench_hepg2_v2_fused.parquet")
d_fus = d_fus[d_fus["gene"].isin(panel) & d_fus["target_gene"].isin(targets)].copy()
print(f"[dEns] fusion delta rows {len(d_fus)}, targets {d_fus['target_gene'].nunique()}", flush=True)

# 3) Δ̂ 级均值（union 语义：单源独有对保留单源值）
both = pd.concat([d_heur.rename(columns={"delta": "d1"}),
                  d_fus.rename(columns={"delta": "d2"})], ignore_index=True)
ens = both.groupby(["target_gene", "gene"], sort=False)[["d1", "d2"]].mean().reset_index()
# union 语义：行向 skipna 均值 = 有几源取几源均值（单源独有对保留单源值）
ens["delta"] = ens[["d1", "d2"]].mean(axis=1, skipna=True)
nboth = int((ens["d1"].notna() & ens["d2"].notna()).sum())
print(f"[dEns] ens rows {len(ens)}, both-source rows {nboth}, "
      f"heur-only {(ens['d1'].notna() & ens['d2'].isna()).sum()}, "
      f"fus-only {(ens['d1'].isna() & ens['d2'].notna()).sum()}", flush=True)
dmap = ens.set_index(["target_gene", "gene"])["delta"]

# 4) pred 构建（local_bench 逐字逻辑）
mask = a.obs["target_gene"].astype(str).to_numpy() == control
Xntc = sp.csr_matrix(a[mask].X)
rng = np.random.default_rng(0)
blocks, pred_perts = [], [control] + targets
for p in pred_perts:
    M = Xntc[rng.choice(Xntc.shape[0], 400, replace=True)].copy()
    if p in dmap.index.get_level_values(0):
        d = dmap.loc[p].reindex(panel).to_numpy()
        f = np.power(2.0, LAM * np.nan_to_num(d, nan=0.0)).astype(np.float32)
        M = M.multiply(f.reshape(1, -1)).tocsr()
        M.data = np.clip(np.rint(M.data), 0, None).astype(np.float32)
        M.eliminate_zeros()
    blocks.append(M)
X = sp.vstack(blocks, format="csr").astype(np.float32)
X.eliminate_zeros()
pred = ad.AnnData(X=X, obs=pd.DataFrame(
    {"target_gene": np.repeat(pred_perts, 400), "context": "Q_HepG2"},
    index=[f"p{i}_c{j}" for i in range(len(pred_perts)) for j in range(400)]))
pred.var_names = panel.to_numpy()
pred.var.index.name = None
outdir = OUT / TAG
outdir.mkdir(parents=True, exist_ok=True)
pp = outdir / "Q_HepG2_pred.h5ad"
pred.write_h5ad(pp, compression="gzip")
print(f"[dEns] wrote {pp} nnz={X.nnz} mean_counts={X.sum()/X.shape[0]:.1f}", flush=True)

# 5) eval + score
ev = outdir / "Q_HepG2_eval"
subprocess.run([str(ENV_BIN / "cell-eval2"), "run", "-ap", str(pp), "-ar", str(REF),
                "--preset", "vcc2026", "--pert-col", "target_gene", "--control", control,
                "--input-type", "counts", "--set", "de.backend=pdex", "-o", str(ev)], check=True)
agg = pd.read_csv(ev / "agg_results.csv")
row = agg[agg["statistic"] == "mean"].iloc[0]
print("[dEns] raw:", {m.split('.')[-1]: round(float(row[m]), 4) for m in row.index
                      if m not in ("statistic",)}, flush=True)
sp_ = outdir / "Q_HepG2_scored.csv"
subprocess.run([str(ENV_BIN / "cell-eval2"), "score", "--user-agg", str(ev / "agg_results.csv"),
                "--real-bundle", str(BUNDLE), "-o", str(sp_)], check=True)
sc = pd.read_csv(sp_).set_index("metric")
print("[dEns] from_replicate:", sc["from_replicate"].to_dict(), flush=True)
