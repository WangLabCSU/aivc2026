"""全深度重建 Orion HCT116 签名（109 批，向量化并行版）。

输出与 build_orion_signatures.py 语义完全一致：每靶 raw-count 求和 → 靶内 CPM 1e4 →
log1p − NTC(log1p CPM 1e4)；gate = 该靶 pass_guide_filter 细胞数 ≥ min_cells。
区别仅在实现：arrow C 级 list-flatten + (靶码,gene) 稳定排序 + 分靶 bincount，
多进程分片累积（partial npz 落盘，父进程 float64 合并），替掉逐细胞 Python 循环。

先备份旧 hct116_all.parquet → hct116_all_3batch.parquet，再原子写新 parquet，
并更新 hct116_build_meta.json（perts/覆盖/NTC 等概况）。

用法: python src/data/build_orion_signatures_v3.py [--min-cells 30] [--workers 10]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
SIG_DIR = ROOT / "data/interim/signatures"
OUT_P = SIG_DIR / "hct116_all.parquet"
OLD_P = SIG_DIR / "hct116_all_3batch.parquet"
META_P = SIG_DIR / "hct116_build_meta.json"
TMP = ROOT / "tmp/sigv3_partials"


def file_sums(fp: str, n_genes: int) -> dict:
    """单批文件 → {target: float64 向量} + NTC 向量/计数 + 每靶细胞数。"""
    f = pq.ParquetFile(fp)
    sums: dict[str, np.ndarray] = {}
    ntc_sum = np.zeros(n_genes, np.float64)
    ntc_n = 0
    cellcnt: dict[str, int] = {}
    for rg in range(f.num_row_groups):
        t = f.read_row_group(
            rg, columns=["gene_token_id", "gene_expression", "gene_target",
                         "pass_guide_filter"])
        t = t.filter(pc.equal(t.column("pass_guide_filter"), 1))
        if t.num_rows == 0:
            continue
        tok = flatten_list_col(t, "gene_token_id").astype(np.int64)
        ex = flatten_list_col(t, "gene_expression").astype(np.float64)
        tgt = t.column("gene_target").to_numpy(zero_copy_only=False)
        if tok.max(initial=-1) >= n_genes:
            raise ValueError(f"{fp}: token id {tok.max()} >= n_genes {n_genes}")
        codes, uniq = pd.factorize(pd.Index(tgt), sort=True)  # 稳定按首现序（细胞级）
        lens = pc.list_value_length(t.column("gene_token_id")).to_numpy(
            zero_copy_only=False).astype(np.int64)
        codes_e = np.repeat(codes, lens)  # 关键: 细胞级码 → 元素级码（逐 cell 展平）
        order = np.argsort(codes_e, kind="stable")
        tok_s, ex_s = tok[order], ex[order]
        code_s = codes_e[order]
        bnd = np.searchsorted(code_s, np.arange(len(uniq) + 1))
        cc = np.bincount(codes, minlength=len(uniq))  # 每靶细胞数（细胞级）
        for i, name in enumerate(uniq):
            s, e = int(bnd[i]), int(bnd[i + 1])
            bc = np.bincount(tok_s[s:e], weights=ex_s[s:e], minlength=n_genes)
            if name == "Non-Targeting":
                ntc_sum += bc
                ntc_n += int(cc[i])
            else:
                cur = sums.get(name)
                if cur is None:
                    sums[name] = bc
                else:
                    cur += bc
                cellcnt[name] = cellcnt.get(name, 0) + int(cc[i])
    return {"sums": sums, "ntc_sum": ntc_sum, "ntc_n": ntc_n, "cellcnt": cellcnt}


def flatten_list_col(table, col: str) -> np.ndarray:
    c = table.column(col)
    if isinstance(c, pa.ChunkedArray):
        c = c.combine_chunks()
    if isinstance(c, pa.ExtensionArray):
        c = c.storage
    return pc.list_flatten(c).to_numpy(zero_copy_only=False)


def worker(args: tuple) -> str:
    files, n_genes, tag = args
    t0 = time.time()
    acc = {"sums": {}, "ntc_sum": None, "ntc_n": 0, "cellcnt": {}}
    for fp in files:
        r = file_sums(fp, n_genes)
        for k, v in r["sums"].items():
            cur = acc["sums"].get(k)
            if cur is None:
                acc["sums"][k] = v
            else:
                cur += v
        acc["ntc_sum"] = r["ntc_sum"] if acc["ntc_sum"] is None else acc["ntc_sum"] + r["ntc_sum"]
        acc["ntc_n"] += r["ntc_n"]
        for k, v in r["cellcnt"].items():
            acc["cellcnt"][k] = acc["cellcnt"].get(k, 0) + v
        print(f"  [w{tag}] {Path(fp).name} perts={len(acc['sums'])} ({time.time()-t0:.0f}s)",
              flush=True)
    names = sorted(acc["sums"])
    mat = np.stack([acc["sums"][n] for n in names]) if names else np.zeros((0, n_genes))
    np.savez(TMP / f"partial_{tag}.npz",
             names=np.array(names), mat=mat,
             ntc_sum=acc["ntc_sum"], ntc_n=np.int64(acc["ntc_n"]),
             cellcnt_names=np.array(sorted(acc["cellcnt"])),
             cellcnt_vals=np.array([acc["cellcnt"][n] for n in sorted(acc["cellcnt"])],
                                   np.int64))
    return f"w{tag}: {len(names)} perts, ntc_n={acc['ntc_n']} ({time.time()-t0:.0f}s)"


def main(min_cells: int, workers: int) -> None:
    t0 = time.time()
    gm = pd.read_parquet(ROOT / "data/public/xatlas/metadata/gene_metadata.parquet")
    tok2sym = gm.set_index("gene_token_id")["gene_name"]
    n_genes = int(gm["gene_token_id"].max()) + 1
    files = sorted(glob.glob(str(ROOT / "data/public/xatlas/data/HCT116_Batch*.parquet")),
                   key=lambda p: int(Path(p).stem.split("Batch")[-1]))
    print(f"{len(files)} batch files, n_genes={n_genes}, workers={workers}", flush=True)
    TMP.mkdir(parents=True, exist_ok=True)
    for old in TMP.glob("partial_*.npz"):
        old.unlink()
    chunks = [(files[i::workers], n_genes, i) for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for msg in ex.map(worker, chunks):
            print(msg, flush=True)

    # ---- 合并 float64 partials
    sums: dict[str, np.ndarray] = {}
    ntc_sum = np.zeros(n_genes, np.float64)
    ntc_n = 0
    cellcnt: dict[str, int] = {}
    for i in range(workers):
        z = np.load(TMP / f"partial_{i}.npz", allow_pickle=False)
        for name, row in zip(z["names"].tolist(), z["mat"]):
            cur = sums.get(name)
            if cur is None:
                sums[name] = row.copy()
            else:
                cur += row
        ntc_sum += z["ntc_sum"]
        ntc_n += int(z["ntc_n"])
        for name, v in zip(z["cellcnt_names"].tolist(), z["cellcnt_vals"].tolist()):
            cellcnt[name] = cellcnt.get(name, 0) + int(v)
    for f_ in TMP.glob("partial_*.npz"):
        f_.unlink()
    n_cells = ntc_n + sum(cellcnt.values())
    print(f"merged: cells={n_cells} (NTC {ntc_n}), perts={len(sums)} ({time.time()-t0:.0f}s)",
          flush=True)

    # ---- 与原脚本一致的 logfc 计算
    ntc_log = np.log1p(ntc_sum / max(ntc_sum.sum(), 1) * 1e4)
    kept = [(t, s) for t, s in sums.items() if s.sum() > 0 and cellcnt.get(t, 0) >= min_cells]
    kept.sort(key=lambda kv: kv[0])
    med_cells = float(np.median([cellcnt[t] for t, _ in kept])) if kept else 0.0
    print(f"kept {len(kept)} perts >= {min_cells} cells (median {med_cells:.0f} cells/靶)",
          flush=True)
    # ---- 哨兵校验（写在 parquet 之前）: 不同靶的签名向量必须互异 + p53 轴方向
    sent = {}
    v0 = kept[0][1]
    cross_diff = max(float(np.abs(s - v0).max()) for _, s in kept[1:min(6, len(kept))])
    cdk_tok = gm.loc[gm["gene_name"] == "CDKN1A", "gene_token_id"].tolist()
    for t_, s_ in kept:
        if t_ in ("ATR", "TP53", "CDKN1A"):
            vals = [round(float(np.log1p(s_[j] / max(s_.sum(), 1) * 1e4) - ntc_log[j]), 4)
                    for j in cdk_tok]
            sent[f"{t_}->CDKN1A"] = vals[0] if len(vals) == 1 else vals
    print(f"sentinels: cross_target_maxdiff={cross_diff:.1f} {sent}", flush=True)
    assert cross_diff > 1.0, "FATAL: 不同靶签名向量近乎相同 — 元素级展开失败"
    genes_arr = tok2sym.reindex(range(n_genes)).to_numpy()
    parts = []
    for tgt, s in kept:
        lf = np.log1p(s / max(s.sum(), 1) * 1e4) - ntc_log
        parts.append(pd.DataFrame({"target_gene": tgt, "gene": genes_arr,
                                   "logfc": lf.astype(np.float32)}))
    out = pd.concat(parts, ignore_index=True)
    del parts

    # ---- 备份旧版 + 原子写新版
    if OUT_P.exists() and not OLD_P.exists():
        shutil.move(str(OUT_P), str(OLD_P))
        print(f"backed up old parquet -> {OLD_P.name}", flush=True)
    tmp_out = OUT_P.with_suffix(".parquet.tmp")
    out.to_parquet(tmp_out)
    os.replace(tmp_out, OUT_P)
    print(f"saved {len(out)} rows, {out['target_gene'].nunique()} perts "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- meta
    comp = pd.read_csv(ROOT / "data/raw/controls/pert_counts.csv")["target_gene"].astype(str)
    cov = sorted(set(comp) & set(out["target_gene"].unique()))
    meta = {"min_cells": min_cells, "perts": int(out["target_gene"].nunique()),
            "rows": int(len(out)), "n_batches": len(files), "n_cells": int(n_cells),
            "ntc_n": ntc_n, "median_cells_per_pert": med_cells,
            "target_cov_300": len(cov),
            "cov_targets": cov,
            "missing_300": sorted(set(comp) - set(cov)),
            "sentinels": {"cross_target_maxdiff_first5": cross_diff, **sent},
            "build": "build_orion_signatures_v3.py (vectorized, 109 batches, float64 merge)"}
    META_P.write_text(json.dumps(meta, indent=1))
    print(f"target_cov_300 = {len(cov)}/300; missing {len(meta['missing_300'])}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    main(a.min_cells, a.workers)
