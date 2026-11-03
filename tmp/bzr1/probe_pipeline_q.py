#!/usr/bin/env python
"""probe_pipeline_q.py — 从 NC bench pred h5ad 提取管道后验 vs sim5 空间的同 target 对齐.

目的: 解释 AMPQ75 bench nmae 爆炸 — pred 值是表达尺度 (nnz 17.2M, max 10502),
而 sim 代理直接测聚合 lfc 签名. 管道在签名与单细胞 pred 之间做了
lambda_real 7.0 混合 + cell_sampling + max_abs_delta/max_fold 帽.
本 probe: 对 sim5 的 100 hold targets, 从 h5ad 取该 target 400 细胞的
pred 矩阵行均值 (去掉 non-targeting), 与 truth lfc 聚合对齐, 算:
  - 每基因 pred 幅度 vs NC/x 幅度的尺度比 → 找出管道的实际幅度行为
  - DE top-50 集 (truth) 上的 |P-Y| 拆解: 过冲部分 (同号 |P|>|Y|) vs 反号
  - pred 行均值的相关/符号一致性 vs NC 签名 → 管道保真度
输出: 三表 (scale ratio 分位, nmae 分解, 每行 per-target 明细 top).
"""
import numpy as np
import pandas as pd
import h5py
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT = "/data/resource/wanglab/aivc/aivc2026"
H5 = f"{ROOT}/outputs/metrics/local_bench/bzr1v11nc_Q/Q_HepG2_pred.h5ad"
CACHE = f"{ROOT}/tmp/bzr1/sim5_cache.npz"
SIG = f"{ROOT}/data/interim/signatures"

# ---- truth aggregate per hold target (sim5 口径: hepg2_all, DE top-50 mask 基于它) ----
z = np.load(CACHE)
X, Y = z["X"], z["Y"]
hold = [s.decode() if isinstance(s, bytes) else str(s) for s in z["hold"]]
n = len(hold)
print(f"[cache] {X.shape} holds={n}", flush=True)

# ---- pred h5ad: per-target row-block mean over genes (sparse rows) ----
f = h5py.File(H5, "r")
obs = f["obs"]
cats = [x.decode() for x in obs["target_gene"]["categories"][()]]
codes = obs["target_gene"]["codes"][()]
var = [x.decode() for x in f["var"]["_index"][()]]
G = len(var)
vardict = {g: i for i, g in enumerate(var)}

indptr = f["X"]["indptr"][()]
data = f["X"]["data"][()]
indices = f["X"]["indices"][()]

# map hold targets -> row ranges
rowcat = {c: [] for c in range(len(cats))}
for i, c in enumerate(codes):
    rowcat.setdefault(c, []).append(i)
# NOTE: codes are per-row category index; build contiguous ranges via cumsum
order = np.argsort(codes, kind="stable")
codes_sorted = codes[order]
starts = np.searchsorted(codes_sorted, np.arange(len(cats)))
ends = np.append(starts[1:], len(codes_sorted))

P = np.zeros((n, G), dtype=np.float32)
found = 0
for j, t in enumerate(hold):
    ci = cats.index(t) if t in cats else -1
    if ci < 0:
        continue
    r0, r1 = starts[ci], ends[ci]
    if r1 <= r0:
        continue
    rows = order[r0:r1]
    # accumulate per-row CSR slice into dense row mean
    acc = np.zeros(G, dtype=np.float32)
    for r in rows:
        s0, s1 = indptr[r], indptr[r + 1]
        idx = indices[s0:s1]
        np.add.at(acc, idx, data[s0:s1])
    P[j] = acc / max(r1 - r0, 1)
    found += 1
print(f"[pred] extracted {found}/{n} targets", flush=True)

# align genes: sim5 matrix columns are 6146 genes — need their names.
# sim5 cache lacks names; rebuild gene list from hct116_all/gwps parquet order.
# Instead: use truth parquet gene order per target via build? Simpler: recompute
# gene universe from the NC signature parquets (6146 = intersection used in sim5).
# We reconstruct from gwps_cal parquet (same gene order as build time).
sigp = f"{SIG}/gwps_calNC.parquet"
if not __import__("os").path.exists(sigp):
    sigp = f"{SIG}/gwps_cal.parquet"
t = pq.read_table(sigp, columns=["gene"])
genes_sig = t.column("gene").unique().to_pylist()
print(f"[sig] genes={len(genes_sig)} from {sigp}", flush=True)

if len(genes_sig) == G:
    gpos = np.arange(G)
else:
    pos = {g: i for i, g in enumerate(var)}
    gpos = np.array([pos[g] for g in genes_sig if g in pos])
print(f"[align] overlapping genes={len(gpos)}", flush=True)

# X/Y columns follow same sig gene order (sim5 built from same parquets)
Pc = P[:, gpos] if gpos.size == X.shape[1] else P[:, gpos[: X.shape[1]]]
Gc = min(Pc.shape[1], X.shape[1])
Pc, Xs, Ys = Pc[:, :Gc], X[:, :Gc], Y[:, :Gc]

mu = Xs.mean(0, keepdims=True)
H = Xs - mu
NC = np.where(Xs * H > 0, H, Xs).astype(np.float32)

# ---- diagnostics ----
den = np.abs(Ys) > np.quantile(np.abs(Ys), 0.995)  # DE proxy (truth big movers)
print(f"\n== scale ratios |P|/max(|NC|,eps) on all genes ==", flush=True)
ratio = np.abs(Pc) / np.maximum(np.abs(NC), 1e-3)
print("quantiles:", {q: round(float(np.quantile(ratio, q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)})

print(f"\n== DE (truth |Y| top-0.5%) |P-Y| decomposition, NC pred ==", flush=True)
err_nc = np.abs(NC - Ys)[den]
overshoot = np.abs(NC - Ys)[den & (np.sign(NC) == np.sign(Ys)) & (np.abs(NC) > np.abs(Ys))]
wrongsign = np.abs(NC - Ys)[den & (np.sign(NC) != np.sign(Ys))]
print(f"total={err_nc.mean():.4f} overshoot_share={overshoot.mean()/max(err_nc.mean(),1e-9)*100:.1f}% "
      f"wrongsign_share={wrongsign.mean()/max(err_nc.mean(),1e-9)*100:.1f}% "
      f"(n_over={len(overshoot)}, n_wrong={len(wrongsign)})")

print(f"\n== pred-vs-NC sign consistency (all genes, per cell-block) ==", flush=True)
same = float(np.mean(np.sign(Pc) == np.sign(NC)))
print(f"sign same fraction: {same:.4f}")

# per-target nmae top/bottom to find systematic offenders
nmae_rows = np.abs(NC - Ys).mean(1)
den_rows = den.mean(1) > 0
nmae_d = np.array([nmae_rows[i] / max(np.abs(Ys)[i][den[i]].mean(), 1e-9) if den[i].any() else np.nan
                   for i in range(n)])
idx = np.argsort(-np.nan_to_num(nmae_d))[:10]
print("\n== worst 10 targets by per-target nmae (NC pred) ==")
for i in idx:
    print(f"  {hold[i]:12s} nmae_row={nmae_d[i]:.3f} |Y|mean_DE={np.abs(Ys)[i][den[i]].mean() if den[i].any() else 0:.3f}")
