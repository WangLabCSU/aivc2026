#!/usr/bin/env python
"""hold_sim8.py — 从 NC bench pred 反推真实 pred lfc 空间, 测候选构造的 fid 机制.

已证: 简单 lfc-cap 等变 (sim7) 无法复现 AMPQ75 的 bench fid -0.0324 (代理仍 dF=0).
bench fid = k/max(n_pred,n_conf), scored set = {p_adj<α}∩denom, 依赖 pdex Wilcoxon
在 400 cells 上的显著性 — 由 NTC 分布形状 × 2^(λv) shift 决定.

本 sim8: 用 NC bench 的 pred counts (h5ad) + ntc_mean profile 反推每基因
"有效 lfc" = log2(pred_mean / ntc_mean), 作为管道响应函数 R(v) 的样本点.
对每个候选构造 v', 预测 pred_lfc(v') ≈ R(v') via 管道一致性:
  实际管道: factor = 2^(λ·v), pred_count = ntc_count × min(factor, fold_cap/1)
  → pred_lfc = log2(pred_count/ntc_count) = min(λ·v, log2(fold_cap̃))
其中 fold_cap 逐基因 = max_fold × orig/orig = 常数 log2(14)=3.807 (对 base>0 基因).
所以 R(v) = min(λ·v, 3.807) — 与 sim7 相同! 差异只在: base=0 基因 (ntc_mean=0)
factor 无效 (0×f=0), pred lfc undefined → 不进显著集.

真正的失真源假设: n_pred 杠杆. AMP 放大后更多基因 |λv| 大 → Wilcoxon p_adj<α
的基因更多 → scored set 变大但 k=50 固定 → fid = k/max(n_pred,n_conf) 下降!
即 fid 是 n_pred 的减函数 — 放大推进的基因多数是"错号"基因 (NC 中心化不改善
真实方向), 它们进显著集只增分母.
sim8 检验: 用每基因显著性代理 prob_sig(g) = sigmoid((|λv| - θ_g)/s_g),
θ_g ∝ NTC 方差 (从 pred h5ad 的 per-pert count 分布估), n_pred = Σ prob_sig,
fid_sim = 50·agree_rate / max(n_pred, n_conf).

输出: 每候选 (pds_sim, fid_sim, n_pred_sim, nmae_sim).
"""
import numpy as np
import h5py
from scipy.stats import rankdata

CACHE = "/data/resource/wanglab/aivc/aivc2026/tmp/bzr1/sim5_cache.npz"
NTC = "/data/resource/wanglab/aivc/aivc2026/tmp/bzr1/ntc_mean_q.npy"
H5 = "/data/resource/wanglab/aivc/aivc2026/outputs/metrics/local_bench/bzr1v11nc_Q/Q_HepG2_pred.h5ad"
LAM = 7.0
CAP = np.log2(14.0)
K = 50
ALPHA = 0.05  # nominal; 用相对阈值校准


def de_mask(Y, k=50):
    n, G = Y.shape
    ath = np.partition(np.abs(Y), -k, axis=1)[:, -k][:, None]
    return np.abs(Y) >= ath


def main():
    z = np.load(CACHE)
    X, Y = z["X"], z["Y"]
    hold = [s.decode() if isinstance(s, bytes) else str(s) for s in z["hold"]]
    n, G = X.shape

    mu = X.mean(0, keepdims=True)
    H = (X - mu).astype(np.float32)
    NC = np.where(X * H > 0, H, X).astype(np.float32)

    # ---- 从 h5ad 反推 per-gene NTC 方差 (count 尺度, per-pert 400 cells 的列方差) ----
    f = h5py.File(H5, "r")
    obs = f["obs"]
    cats = [x.decode() for x in obs["target_gene"]["categories"][()]]
    codes = f["obs"]["target_gene"]["codes"][()]
    varnames = [x.decode() for x in f["var"]["_index"][()]]
    ntc = cats.index("non-targeting")
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(len(cats)))
    ends = np.append(starts[1:], len(codes))
    rows_ntc = order[starts[ntc]:ends[ntc]]
    # 取前 100 个 NTC 细胞估列方差 (节省内存)
    rows_ntc = rows_ntc[:100]
    ntc_cells = np.zeros((len(rows_ntc), len(varnames)), dtype=np.float32)
    Xf, ip, ix = f["X"], f["X"]["indptr"][()], f["X"]["indices"][()]
    for i, r in enumerate(rows_ntc):
        s0, s1 = ip[r], ip[r + 1]
        ntc_cells[i, ix[s0:s1]] = Xf["data"][s0:s1]
    ntc_mean = ntc_cells.mean(0)
    ntc_sd = ntc_cells.std(0) + 1e-6
    # 相对 CV: 显著性阈值 ∝ CV (低表达基因 counts 小 CV 大)
    cv = ntc_sd_over = ntc_sd = ntc_cells.std(0) / np.maximum(ntc_mean, 0.5)
    np.save("tmp/bzr1/cv_q.npy", cv.astype(np.float32))
    print(f"[cv] median={np.median(cv):.3f} q90={np.quantile(cv,0.9):.3f}", flush=True)

    # ---- 基因对齐: h5ad varnames → sim5 6146 列. sim5 列 = gwps_cal parquet gene 顺序
    # 但 sim5 cache 未存 gene 名. 重建: sim5 由 (gwps+hct116) target_gene 均值对齐
    # 6146 = 面板基因数? gene_names.csv
    import pandas as pd
    panel = pd.read_csv("data/raw/controls/gene_names.csv")["gene_name"].astype(str).tolist()
    print(f"[panel] {len(panel)} genes", flush=True)
    pos = {g: i for i, g in enumerate(varnames)}
    # sim5 矩阵列序未知 — 假设 = panel 序 (sim5 构建自 parquet groupby, 序不可考)
    # 退路: sim5 cache X 的列就是它构建时的基因序 — 与 bench 无关, 只需将 cv 映射到
    # 同一基因. 没有 gene 名就不可能对齐! 放弃列级对齐, 改为分布级: 把 cv 分布
    # 量化为阈值 θ 分布, 每基因随机配对 (近似 — 分布正确即 n_pred 统计正确).
    rng = np.random.default_rng(0)
    cv_sim = rng.choice(cv, size=X.shape[1], replace=True)
    cv_sim = np.sort(cv_sim)  # 表达量与 lfc 幅度负相关: 高表达低变异 → 低 cv ↔ 高 |v|?
    # 更合理: 高表达基因 cv 低, 且高表达基因 lfc 被测得更稳 → |v| 大的基因偏高表达.
    # 用 -|mu_g| 秩 = cv 秩 排列配对
    rank_v = np.argsort(np.argsort(np.abs(X).mean(0)))
    rank_cv = np.argsort(np.argsort(cv))
    cv_sim = np.empty(X.shape[1])
    cv_sim[rank_v] = np.sort(cv)[rank_v * 0 + np.arange(X.shape[1])] if False else cv[np.argsort(rank_cv := rank_cv)]
    print("WARN: 简化 — 用 cv 排序对 |v| 均值排序的逆序配对", flush=True)


if __name__ == "__main__":
    main()
