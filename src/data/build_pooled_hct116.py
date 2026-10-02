"""HCT116 全批次 pooled 重建（Orion xatlas parquet → 单一 h5ad）。流式增量写盘，内存 ~5GB。

背景: 全量 ~670 万细胞全转录组 CSR 的 sp.vstack 会物化巨型 COO（实测 3.4M cells 时
RSS 533GB 被 cgroup OOM 杀死），故直接用 h5py 流式组装 anndata 兼容的 csr_matrix
h5ad（结构镜像 anndata 0.12: root/X/obs/var attrs 逐一核对过）。

输入: data/public/xatlas/data/HCT116_Batch*.parquet（每行=1 细胞，list 列
      gene_token_id/gene_expression + 标量 gene_target/sample/pass_guide_filter）
输出: data/public/xatlas/HCT116_pooled.h5ad（无压缩，原子替换）
      data/public/xatlas/.pooled_full.done  {n_cells, n_batches, n_genes, ...}

幂等: .pooled_full.done 存在 且其 n_batches == 当前批次数 且 n_cells 相符
      且 mtime >= 下载 marker (.orion_all.done) mtime → 跳过重建。

用法: python src/data/build_pooled_hct116.py [--force]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
XDIR = ROOT / "data/public/xatlas"
MARKER = XDIR / ".orion_all.done"
DONE = XDIR / ".pooled_full.done"


def count_cells(fp: str) -> int:
    import pyarrow.parquet as pq
    return int(pq.ParquetFile(fp).metadata.num_rows)


def unique_symbols(sym: np.ndarray) -> np.ndarray:
    """var_names_make_unique 的 numpy 版（重复名追加 -1/-2/...）。"""
    out = sym.astype(str).copy()
    seen: dict[str, int] = Counter()
    for i, s in enumerate(out):
        n = seen[s]
        seen[s] = n + 1
        if n:
            out[i] = f"{s}-{n}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    files = sorted(glob.glob(str(XDIR / "data" / "HCT116_Batch*.parquet")))
    if not files:
        print("[pooled] no batch parquet found", flush=True)
        sys.exit(1)
    total_cells = sum(count_cells(f) for f in files)
    if DONE.exists() and not args.force:
        try:
            d = json.loads(DONE.read_text())
            marker_ok = (not MARKER.exists()) or DONE.stat().st_mtime >= MARKER.stat().st_mtime
            if (d.get("n_batches") == len(files) and d.get("n_cells") == total_cells
                    and marker_ok):
                print(f"[pooled] up-to-date skip: {d}", flush=True)
                return
        except Exception as e:  # noqa: BLE001
            print(f"[pooled] done-file unreadable ({e}), rebuilding", flush=True)

    import anndata as ad
    import h5py
    import pyarrow.parquet as pq

    gm = pd.read_parquet(XDIR / "metadata" / "gene_metadata.parquet")
    n_genes = int(gm["gene_token_id"].max()) + 1
    sym = unique_symbols(gm.set_index("gene_token_id")["gene_name"]
                         .reindex(range(n_genes)).to_numpy())
    print(f"[pooled] {len(files)} batches, {total_cells} cells, {n_genes} genes", flush=True)

    out = XDIR / "HCT116_pooled.h5ad"
    tmp = XDIR / "HCT116_pooled.h5ad.tmp"
    if tmp.exists():
        tmp.unlink()
    gts: list[str] = []
    samples: list[str] = []
    indptr_acc = [0]
    nnz_total = 0
    r = 0
    with h5py.File(tmp, "w") as f:
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"
        for g in ("layers", "obsm", "obsp", "uns", "varm", "varp"):
            gg = f.create_group(g)
            gg.attrs["encoding-type"] = "dict"
            gg.attrs["encoding-version"] = "0.1.0"
        Xg = f.create_group("X")
        Xg.attrs["encoding-type"] = "csr_matrix"
        Xg.attrs["encoding-version"] = "0.1.0"
        d_data = Xg.create_dataset("data", shape=(0,), maxshape=(None,),
                                   dtype="float32", chunks=(1 << 20,))
        d_idx = Xg.create_dataset("indices", shape=(0,), maxshape=(None,),
                                  dtype="int32", chunks=(1 << 20,))
        for fi, fp in enumerate(files):
            pf = pq.ParquetFile(fp)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg).to_pandas()
                t = t[t["pass_guide_filter"] == 1]
                if len(t) == 0:
                    continue
                lens = t["gene_token_id"].map(len).to_numpy(np.int64)
                rows = np.repeat(np.arange(len(t), dtype=np.int32), lens)
                cols = np.concatenate(t["gene_token_id"].to_numpy()).astype(np.int32)
                vals = np.concatenate(t["gene_expression"].to_numpy()).astype(np.float32)
                M = sp.coo_matrix((vals, (rows, cols)), shape=(len(t), n_genes)).tocsr()
                M.data = np.rint(M.data)
                M.sum_duplicates()
                M = M.astype(np.float32)
                M.eliminate_zeros()
                n = M.nnz
                d_data.resize(nnz_total + n, axis=0)
                d_idx.resize(nnz_total + n, axis=0)
                d_data[nnz_total:nnz_total + n] = M.data
                d_idx[nnz_total:nnz_total + n] = M.indices.astype(np.int32)
                indptr_acc.extend((M.indptr[1:].astype(np.int64) + nnz_total).tolist())
                nnz_total += n
                r += len(t)
                gts.extend(t["gene_target"].astype(str).tolist())
                samples.extend(t["sample"].astype(str).tolist())
                del M
            print(f"[pooled] file {fi+1}/{len(files)}, cells={r}, nnz={nnz_total} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if (fi + 1) % 25 == 0:
                gc_collect()
        indptr = np.asarray(indptr_acc, dtype=np.int64)
        assert len(indptr) == r + 1
        Xg.create_dataset("indptr", data=indptr, dtype="int64")
        Xg.attrs["shape"] = np.array([r, n_genes], dtype=np.int64)
        obs = pd.DataFrame({
            "target_gene": pd.Categorical(gts),
            "sample": pd.Categorical(samples),
        }, index=pd.RangeIndex(r))
        var = pd.DataFrame(index=pd.Index(sym, name=None))
        ad.io.write_elem(f, "obs", obs)
        ad.io.write_elem(f, "var", var)
    os.replace(tmp, out)
    vc = pd.Series(gts).value_counts().drop("Non-Targeting", errors="ignore")
    stats = {"n_cells": r, "n_batches": len(files), "n_genes": n_genes,
             "nnz": int(nnz_total), "n_ntc": int((pd.Series(gts) == "Non-Targeting").sum()),
             "perts_ge30": int((vc >= 30).sum()),
             "median_cells_per_pert": float(vc.median()),
             "seconds": round(time.time() - t0, 1)}
    DONE.write_text(json.dumps(stats, indent=1))
    print(f"[pooled] saved {out}: {stats}", flush=True)


def gc_collect() -> None:
    import gc
    gc.collect()


if __name__ == "__main__":
    main()
