"""v3.6: 从 plate12.h5ad 提取全部 50 细胞系的 DMSO_TF basal（vehicle 对照画像）。

输出: data/interim/l15_fusion/v36/raw_basal_{CVCL}.npy  (62,710, float64, plate12 var 序)
约定: raw = 该系 DMSO_TF 细胞 X 列和（sum）；build_aux 再做 CPM→log1p（sum/mean 在
CPM 下等价，与 v3.5 raw_basal_a549.npy 约定一致）。
单次顺序扫描 37.5GB CSR，逐块 one-hot 投影累加，内存峰值 ~6GB。

用法: python scripts/l15fusion_v36_basal.py [--chunk 100000]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
H5AD = ROOT / "data/public/tahoe/plate12.h5ad"
OUT = ROOT / "data/interim/l15_fusion/v36"
VEHICLE = "DMSO_TF"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=100_000)
    args = ap.parse_args()
    import anndata as ad

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    a = ad.read_h5ad(H5AD, backed="r")
    n, g = a.shape
    cl = a.obs["cell_line"].astype(str).to_numpy()
    dr = a.obs["drug"].astype(str).to_numpy()
    lines = sorted(set(cl))
    print(f"{n} cells x {g} vars, {len(lines)} cell lines, "
          f"DMSO_TF rows={int((dr == VEHICLE).sum())} ({time.time()-t0:.0f}s)", flush=True)
    lid = {x: i for i, x in enumerate(lines)}
    acc = np.zeros((len(lines), g), np.float64)
    cnt = np.zeros(len(lines), np.int64)
    isv = dr == VEHICLE
    CH = args.chunk
    for s in range(0, n, CH):
        e = min(s + CH, n)
        sel = np.where(isv[s:e])[0]
        if len(sel) == 0:
            continue
        Xd = sp.csr_matrix(a.X[s + sel])          # DMSO 行子块
        gcodes = np.array([lid[x] for x in cl[s + sel]], np.int64)
        M = sp.csr_matrix((np.ones(len(sel), np.float32),
                           (np.arange(len(sel)), gcodes)), shape=(len(sel), len(lines)))
        acc = acc + np.asarray((M.T @ Xd).astype(np.float64))  # one-hot 投影 → (50, g) 列和(asarray 防 np.matrix)
        cnt += np.bincount(gcodes, minlength=len(lines))
        if (s // CH) % 20 == 0:
            print(f"  scan {e}/{n} dmso_acc={int(cnt.sum())} ({time.time()-t0:.0f}s)", flush=True)
    a.file.close()
    meta = {}
    for i, x in enumerate(lines):
        p = OUT / f"raw_basal_{x}.npy"
        np.save(p, acc[i].astype(np.float64))
        meta[x] = {"n_dmso_cells": int(cnt[i]), "sum_counts": float(acc[i].sum()),
                   "nonzero_genes": int((acc[i] > 0).sum())}
    (OUT / "basal_extract_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {len(lines)} raw_basal_*.npy; cells/line min={cnt.min()} max={cnt.max()} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
