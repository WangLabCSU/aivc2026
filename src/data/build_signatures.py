"""构建 CRISPRi 伪批量 logFC 签名库（L1 主监督）。

对每个 scPerturb 数据集：按靶基因伪批量（raw counts 求和 → CP10k → log1p），
logFC = log1p(pert) - log1p(NTC均值)。只保留竞赛 300 靶基因 + 全局均值签名所需数据。
输出 data/interim/signatures/<name>.parquet: 列 [target_gene, gene, logfc, n_cells]。

用法: python src/data/build_signatures.py [--datasets gwps,norman]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
SCP = ROOT / "data/public/scperturb"
OUT = ROOT / "data/interim/signatures"

DATASETS = {
    "gwps": ("Replogle2022_K562_gwps.h5ad", "gene"),
    "k562_ess": ("Replogle2022_K562_essential.h5ad", "gene"),
    "rpe1": ("Replogle2022_rpe1.h5ad", "gene"),
    "hepg2": ("Nadig2024_hepg2.h5ad", "gene"),
    "jurkat": ("Nadig2024_jurkat.h5ad", "gene"),
    "norman": ("NormanWeissman2019.h5ad", "perturbation"),
    "adamson": ("AdamsonWeissman2016.h5ad", "perturbation"),
}
CONTROL_VALUES = {"non-targeting", "ntc", "control", "safe-targeting"}


def pseudobulk_logfc(path: Path, pert_col: str, wanted: set[str] | None,
                     chunk_cells: int = 400_000) -> pd.DataFrame:
    """分块读 h5ad，指示矩阵稀疏乘一次求全组伪批量；返回 logFC 长表。

    wanted=None → 保留全部非对照扰动（用于 gwps 全量构建通用兜底签名）。
    """
    a = ad.read_h5ad(path, backed="r")
    pert = pd.Series(a.obs[pert_col].astype(str).to_numpy())
    genes = pd.Index(a.var_names.astype(str))
    ntc_mask = (pert.str.contains("non-targeting|safe", case=False, regex=True)
                | pert.isin(CONTROL_VALUES)).to_numpy()

    all_p = pert.unique()
    is_ctrl = (pd.Series(list(all_p)).str.contains("non-targeting|safe", case=False,
                                                   regex=True)
               | pd.Series(list(all_p)).isin(CONTROL_VALUES)).to_numpy()
    keep = all_p[~is_ctrl] if wanted is None else all_p[~is_ctrl & pd.Series(list(all_p)).isin(wanted).to_numpy()]
    pidx = {p: i for i, p in enumerate(keep)}
    sums = np.zeros((len(keep), a.n_vars), dtype=np.float64)
    ntc_sum = np.zeros(a.n_vars, dtype=np.float64)
    ntc_n = 0
    for start in range(0, a.n_obs, chunk_cells):
        stop = min(start + chunk_cells, a.n_obs)
        X = sp.csr_matrix(a[start:stop].X)
        p_chunk = pert.iloc[start:stop].to_numpy()
        m_chunk = ntc_mask[start:stop]
        # 指示矩阵（cells × n_perts）一次乘出全组求和（向量化列映射）
        cols = pd.Series(p_chunk).map(pidx).to_numpy()
        sel = ~pd.isna(cols)
        g_rows = np.nonzero(sel)[0]
        g_cols = cols[sel].astype(np.int64)
        G = sp.csr_matrix(
            (np.ones(len(g_rows)), (g_rows, g_cols)),
                shape=(stop - start, len(keep)))
        sums += (G.T @ X).toarray()
        if m_chunk.any():
            ntc_sum += np.asarray(X[m_chunk].sum(axis=0)).ravel()
            ntc_n += int(m_chunk.sum())
        print(f"  chunk {stop}/{a.n_obs}", flush=True)
    a.file.close()

    def to_log1p_cp10k(v: np.ndarray) -> np.ndarray:
        return np.log1p(v / max(v.sum(), 1.0) * 1e4)

    ntc_log = to_log1p_cp10k(ntc_sum)
    ntc_expr = ntc_sum / max(ntc_n, 1)  # 平均 UMI，用于过滤未表达基因
    rows = []
    for j, p in enumerate(keep):
        lf = to_log1p_cp10k(sums[j]) - ntc_log
        df = pd.DataFrame({
            "target_gene": p,
            "gene": genes.to_numpy(),
            "logfc": lf.astype(np.float32),
            "ntc_mean_umi": ntc_expr.astype(np.float32),
        })
        df = df[(df["ntc_mean_umi"] > 0.05) & (df["logfc"].abs() > 1e-4)]
        rows.append(df[["target_gene", "gene", "logfc"]])
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["target_gene", "gene", "logfc"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="gwps,norman,k562_ess,rpe1,hepg2,jurkat")
    ap.add_argument("--all-perts", action="store_true",
                    help="wanted=None 全量重建（全部非对照扰动；gwps 恒为全量）")
    ap.add_argument("--suffix", default="",
                    help="输出文件名后缀（如 _all → norman_all.parquet），避免覆盖 300 靶版")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    targets = set(pd.read_csv(ROOT / "data/raw/controls/pert_counts.csv")["target_gene"])
    # 全局均值签名需要所有扰动，不仅 300 靶
    want_all = args.datasets.split(",")

    for name in want_all:
        fname, col = DATASETS[name]
        print(f"[{name}] building ...", flush=True)
        wanted = None if (name == "gwps" or args.all_perts) else targets
        df = pseudobulk_logfc(SCP / fname, col, wanted)
        df.to_parquet(OUT / f"{name}{args.suffix}.parquet")
        hits = targets & set(df["target_gene"].unique()) if len(df) else set()
        print(f"[{name}] {len(df)} rows, {df['target_gene'].nunique() if len(df) else 0} perts, "
              f"target coverage {len(hits)}/300", flush=True)


if __name__ == "__main__":
    main()
