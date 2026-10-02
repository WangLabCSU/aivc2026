"""从 Orion HCT116 多批次 parquet 构建伪批量 logFC 签名（p53-WT CRC 训练上下文）。

合并所有可用批次 → 每靶聚齐 ≥30 细胞的靶进入签名（gate 内体积充足）。
用法: python src/data/build_orion_signatures.py [--min-cells 30]
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]

def main(min_cells: int):
    gm = pd.read_parquet(ROOT / 'data/public/xatlas/metadata/gene_metadata.parquet')
    tok2sym = gm.set_index('gene_token_id')['gene_name']
    n_genes = int(gm['gene_token_id'].max()) + 1
    files = sorted(glob.glob(str(ROOT / 'data/public/xatlas/data/HCT116_Batch*.parquet')))
    print(f'{len(files)} batch files', flush=True)
    sums, ntc_sum, ntc_n, cellcnt = {}, np.zeros(n_genes), 0, {}
    for fp in files:
        f = pq.ParquetFile(fp)
        for rg in range(f.num_row_groups):
            t = f.read_row_group(rg, columns=['gene_token_id','gene_expression','gene_target','pass_guide_filter']).to_pandas()
            t = t[t['pass_guide_filter'] == 1]
            for tid, ex, tgt in zip(t['gene_token_id'], t['gene_expression'], t['gene_target']):
                v = np.zeros(n_genes); v[list(tid)] = ex
                if tgt == 'Non-Targeting':
                    ntc_sum += v; ntc_n += 1
                else:
                    s = sums.setdefault(tgt, np.zeros(n_genes)); s += v
                    cnts = cellcnt.setdefault(tgt, 0); cellcnt[tgt] = cnts + 1
        print(f'  {fp.split("/")[-1]} done (ntc_n={ntc_n}, perts={len(sums)})', flush=True)
    ntc_log = np.log1p(ntc_sum / max(ntc_sum.sum(), 1) * 1e4)
    rows = []
    for tgt, s in sums.items():
        if s.sum() <= 0 or cellcnt.get(tgt, 0) < min_cells: continue
        # 细胞数估计: 无逐靶计数列, 用该靶 sum/中位单细胞 total 近似 -> 改用文件级聚合时记录
        lf = np.log1p(s / max(s.sum(), 1) * 1e4) - ntc_log
        df = pd.DataFrame({'target_gene': tgt, 'gene': tok2sym.reindex(range(n_genes)).to_numpy(), 'logfc': lf.astype(np.float32)})
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(ROOT / 'data/interim/signatures/hct116_all.parquet')
    print(f'saved {len(out)} rows, {out["target_gene"].nunique()} perts', flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--min-cells', type=int, default=30); a = ap.parse_args(); main(a.min_cells)
