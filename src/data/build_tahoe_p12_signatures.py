"""Tahoe plate12 药扰伪批量签名构建（fusion v3 辅助任务资产）。

从 plate12 h5ad（A549+SW480 上皮系，380 药）构建 cell_line × drug 的伪批量 logFC
签名（vs 同板同系 DMSO 对照）。输出 data/interim/signatures/tahoe_p12.parquet，
列 [target_gene(=药名_剂量), gene, logfc]——target_gene 字段承载"扰动"语义以复用
fusion 特征管线（药扰作为辅助训练任务，非竞赛靶语义）。

用法: python src/data/build_tahoe_p12_signatures.py [--chunk 200000]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/public/tahoe/plate12.h5ad"
OUT = ROOT / "data/interim/signatures/tahoe_p12.parquet"
CONTROL_TOKEN = "DMSO"  # 药物载体对照


def main(chunk: int):
    a = ad.read_h5ad(SRC, backed="r")
    cl = a.obs["cell_line"].astype(str).to_numpy()
    drug = a.obs["drug"].astype(str).to_numpy()
    is_ctrl = np.array(["DMSO" in d or "dmso" in d for d in drug])
    genes = pd.Index(a.var_names.astype(str))
    n_genes = a.n_vars
    print(f"plate12: {a.n_obs} cells × {n_genes} genes | DMSO cells: {int(is_ctrl.sum())}", flush=True)

    # 组键: cell_line + drug（剂量并入键）
    conc = a.obs["drugname_drugconc"].astype(str).to_numpy()
    key = np.array([f"{c}|{d}" for c, d in zip(cl, drug)])
    ctrl_keys = sorted({k for k, m in zip(key, is_ctrl) if m})
    print(f"groups: {len(set(key))} (DMSO groups: {len(ctrl_keys)})", flush=True)

    sums: dict[str, np.ndarray] = {}
    ntc = {}
    for start in range(0, a.n_obs, chunk):
        stop = min(start + chunk, a.n_obs)
        X = sp.csr_matrix(a[start:stop].X)
        k_chunk = key[start:stop]
        for j in range(stop - start):
            k = k_chunk[j]
            if k not in sums:
                sums[k] = np.zeros(n_genes)
            sums[k] += np.asarray(X[j].todense()).ravel()
        print(f"  chunk {stop}/{a.n_obs}", flush=True)
    a.file.close()

    rows = []
    for ck in ctrl_keys:
        cl_name = ck.split("|")[0]
        ntc_log = np.log1p(sums[ck] / max(sums[ck].sum(), 1) * 1e4)
        for k, s in sums.items():
            if k.startswith(cl_name + "|") and "DMSO" not in k:
                lf = np.log1p(s / max(s.sum(), 1) * 1e4) - ntc_log
                drugname = k.split("|")[1]
                df = pd.DataFrame({
                    "target_gene": f"{cl_name}|{drugname}",
                    "gene": genes.to_numpy(),
                    "logfc": lf.astype(np.float32)})
                rows.append(df[df["logfc"].abs() > 0.05])
    out = pd.concat(rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    print(f"saved {len(out)} rows, {out['target_gene'].nunique()} drug-groups", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=200_000)
    a = ap.parse_args()
    main(a.chunk)
