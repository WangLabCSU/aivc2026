"""HCT116 (xatlas 3batches) 伪批量 logFC 签名：p53-WT 结直肠癌补位上下文。

数据: data/public/xatlas/HCT116_3batches.h5ad (56,655 cells × 38,606 genes,
obs.target_gene 含 'Non-Targeting' 2,782 cells; 3 batches)。
与 build_signatures.pseudobulk_logfc 同口径: CP10K log1p(pert) − log1p(NTC)，
过滤 ntc_mean_umi>0.05 且 |logfc|>1e-4；额外:
  - 逐 pert 记录 n_cells，min_cells 阈值过滤（xatlas 每 pert 细胞数极稀），
    并打印分层方向一致率 vs gwps 供阈值决策留档
  - 丢弃 var_names 中 'ENSG...' 兜底 id（无符号、无面板匹配价值）
输出: data/interim/signatures/hct116_all.parquet [target_gene, gene, logfc, n_cells]

用法: python src/data/build_signatures_hct116.py [--min-cells 5]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
H5AD = ROOT / "data/public/xatlas/HCT116_3batches.h5ad"
OUT = ROOT / "data/interim/signatures/hct116_all.parquet"
CONTROL = "non-targeting"  # 大小写不敏感匹配 'Non-Targeting'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cells", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    a = ad.read_h5ad(H5AD, backed="r")
    pert = pd.Series(a.obs["target_gene"].astype(str).to_numpy())
    batch = pd.Series(a.obs["sample"].astype(str).to_numpy())
    genes_all = pd.Index(a.var_names.astype(str))
    keep_var = ~genes_all.str.startswith("ENSG")
    genes = genes_all[keep_var].to_numpy()
    print(f"h5ad {a.n_obs}x{a.n_vars}, drop {int((~keep_var).sum())} ENSG-id vars",
          flush=True)

    is_ntc = pert.str.contains(CONTROL, case=False, regex=False).to_numpy()
    all_p = pert.unique()
    keep_p = np.array([p for p in all_p
                       if not p.lower().startswith("non-targeting")], dtype=object)
    pidx = {p: i for i, p in enumerate(keep_p)}
    n_p = len(keep_p)
    sums = np.zeros((n_p, len(genes)), np.float64)
    ncells = np.zeros(n_p, np.int32)
    ntc_sum = np.zeros(len(genes), np.float64)
    ntc_n = 0
    ntc_by_batch = {}
    CH = 20_000
    for s in range(0, a.n_obs, CH):
        stop = min(s + CH, a.n_obs)
        X = sp.csr_matrix(a[s:stop].X)[:, keep_var]
        p_ch = pert.iloc[s:stop].to_numpy()
        b_ch = batch.iloc[s:stop].to_numpy()
        cols = pd.Series(p_ch).map(pidx).to_numpy()
        sel = ~pd.isna(cols)
        gr = np.nonzero(sel)[0]
        gc = cols[sel].astype(np.int64)
        G = sp.csr_matrix((np.ones(len(gr)), (gr, gc)), shape=(stop - s, n_p))
        sums += (G.T @ X).toarray()
        np.add.at(ncells, gc, 1)
        m = pd.Series(p_ch).str.contains(CONTROL, case=False, regex=False).to_numpy()
        if m.any():
            Xn = X[m]
            ntc_sum += np.asarray(Xn.sum(axis=0)).ravel()
            ntc_n += int(m.sum())
            for b in np.unique(b_ch[m]):
                mb = m & (b_ch == b)
                ntc_by_batch[b] = ntc_by_batch.get(b, 0) + int(mb.sum())
        if (s // CH) % 1 == 0:
            print(f"  chunk {stop}/{a.n_obs} ({time.time()-t0:.0f}s)", flush=True)
    a.file.close()
    print(f"NTC n={ntc_n} by batch {ntc_by_batch}", flush=True)

    def cp10k_log1p(v: np.ndarray) -> np.ndarray:
        return np.log1p(v / max(v.sum(), 1.0) * 1e4)

    ntc_log = cp10k_log1p(ntc_sum)
    ntc_expr = ntc_sum / max(ntc_n, 1)
    dfs = []
    for j, p in enumerate(keep_p):
        if ncells[j] < 2:
            continue
        lf = cp10k_log1p(sums[j]) - ntc_log
        df = pd.DataFrame({"target_gene": p, "gene": genes, "logfc": lf.astype(np.float32),
                           "n_cells": np.int32(ncells[j])})
        df = df[(ntc_expr > 0.05) & (df["logfc"].abs() > 1e-4)]
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    print(f"full (>=2 cells): {len(full)} rows, {full.target_gene.nunique()} perts "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 阈值决策留档: 分层方向一致率 vs gwps（|logfc|>=0.25 的高幅基因）
    gw = pd.read_parquet(ROOT / "data/interim/signatures/gwps.parquet",
                         columns=["target_gene", "gene", "logfc"])
    gw = gw.groupby(["target_gene", "gene"], sort=False)["logfc"].mean()
    strata = {"3-4": (2, 5), "5-9": (5, 10), ">=10": (10, 10**9)}
    agree = {}
    for name, (lo, hi) in strata.items():
        sub = full[(full.n_cells >= lo) & (full.n_cells < hi)]
        m = sub.merge(gw.rename("gwps"), left_on=["target_gene", "gene"],
                      right_index=True, how="inner")
        hi_g = m[m.gwps.abs() >= 0.25]
        agr = float((np.sign(hi_g.logfc) == np.sign(hi_g.gwps)).mean()) if len(hi_g) else np.nan
        # 幅度衰减: 相对幅度 (hct116 logfc 幅 / gwps 幅) 中位数
        att = float((hi_g.logfc.abs() / hi_g.gwps.abs().clip(lower=0.25)).median()) if len(hi_g) else np.nan
        agree[name] = {"perts": int(sub.target_gene.nunique()), "agr": round(agr, 3),
                       "att": round(att, 3)}
        print(f"  [{name}] perts={agree[name]['perts']} dir_agr_vs_gwps={agr:.3f} "
              f"amp_ratio_med={att:.2f}", flush=True)

    sig = full[full.n_cells >= args.min_cells].copy()
    print(f"min_cells={args.min_cells}: {len(sig)} rows, {sig.target_gene.nunique()} perts",
          flush=True)
    targets = set(pd.read_csv(ROOT / "data/raw/controls/pert_counts.csv")["target_gene"])
    print(f"  competition 300-target coverage: {len(targets & set(sig.target_gene.unique()))}/300",
          flush=True)
    sig[["target_gene", "gene", "logfc", "n_cells"]].to_parquet(OUT)

    # ---- p53 / IFN 模块方向证据
    mods = {
        "p53": ["CDKN1A", "BBC3", "MDM2", "GADD45A", "DDB2", "SESN1", "PMAIP1", "BAX",
                "ZMAT3", "EI24"],
        "ifn": ["ISG15", "IFIT1", "IFIT3", "IFI6", "IFI44L", "OAS1", "MX1", "IRF7",
                "RSAD2", "CXCL10"],
    }
    for tgt in ["TP53", "CDKN1A", "MDM2"]:
        sub = sig[sig.target_gene == tgt]
        row = {}
        for mn, gl in mods.items():
            v = sub[sub.gene.isin(gl)].logfc.mean()
            row[mn] = round(float(v), 3)
        print(f"  module-mean logfc @{tgt}: {row}", flush=True)
    # NTC 基线 TP53 轴活性（对照 K562 gwps NTC 无法直接比，用 TP53 靶效应方向即证据）
    (ROOT / "data/interim/signatures/hct116_build_meta.json").write_text(pd.Series(
        {**{f"agree_{k}": str(v) for k, v in agree.items()},
         "min_cells": args.min_cells, "perts": int(sig.target_gene.nunique()),
         "rows": int(len(sig)), "ntc_n": ntc_n,
         "target_cov_300": len(targets & set(sig.target_gene.unique()))}).to_json())
    print(f"[done] → {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
