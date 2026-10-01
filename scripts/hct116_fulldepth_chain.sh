#!/bin/bash
# HCT116 全深度链：等下载 → pooled 重建 → ref → λ1.5/fusion v1/v2 评分
# 用法: nohup bash scripts/hct116_fulldepth_chain.sh > outputs/logs/hct116_chain.log 2>&1 &
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
B=/data/home/wsx/miniconda3/envs/aivc2026/bin/cell-eval2
LOG=outputs/logs/hct116_chain.log

echo "[chain] waiting for .orion_all.done ..."
while [ ! -f data/public/xatlas/.orion_all.done ]; do sleep 120; done
echo "[chain] download complete $(date +%T)"

# 1) pooled 重建（全部批次）
$PY - <<'PYEOF'
import glob, pyarrow.parquet as pq
import pandas as pd, numpy as np, scipy.sparse as sp, anndata as ad
gm = pd.read_parquet('data/public/xatlas/metadata/gene_metadata.parquet')
n_genes = int(gm['gene_token_id'].max()) + 1
sym = gm.set_index('gene_token_id')['gene_name'].reindex(range(n_genes)).to_numpy()
files = sorted(glob.glob('data/public/xatlas/data/HCT116_Batch*.parquet'))
print(f'{len(files)} batches', flush=True)
rows, cols, vals, gts, samples = [], [], [], [], []
r = 0
for fp in files:
    f = pq.ParquetFile(fp)
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg).to_pandas()
        t = t[t['pass_guide_filter'] == 1]
        for tid, ex, tgt, smp in zip(t['gene_token_id'], t['gene_expression'], t['gene_target'], t['sample']):
            rows.extend([r]*len(tid)); cols.extend(list(tid)); vals.extend(list(ex)); r += 1
            gts.append(tgt); samples.append(smp)
X = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(r, n_genes))
X.data = np.rint(X.data)
obs = pd.DataFrame({'target_gene': gts, 'sample': samples})
a = ad.AnnData(X=X, obs=obs)
a.var_names = sym; a.var_names_make_unique()
a.write_h5ad('data/public/xatlas/HCT116_pooled.h5ad')
vc = obs['target_gene'].value_counts().drop('Non-Targeting', errors='ignore')
print('pooled saved:', a.shape, '| NTC:', int((obs.target_gene=="Non-Targeting").sum()),
      '| perts>=400:', int((vc>=400).sum()), '| median:', float(vc.median()), flush=True)
PYEOF
[ $? -ne 0 ] && { echo "[chain] pooled build FAILED"; exit 1; }

# 2) 参考（gate 400: 每扰动 400 细胞）
rm -rf data/interim/local_bench/S_HCT116
$PY src/eval/local_bench.py --model-config configs/l1_signature/ws-v1.2-sampled-lam15.yaml --tag l11_lam1.5 --contexts S_HCT116 > outputs/logs/chain_l15.log 2>&1
echo "[chain] lambda1.5 scored (see outputs/metrics/local_bench/l11_lam1.5/S_HCT116_eval)"

# 3) 归一化 bundle（nmae 此时应有值）
$PY src/eval/bench_score.py --tags l11_lam1.5 --contexts S_HCT116 --build-bundles > outputs/logs/chain_bundle.log 2>&1
echo "[chain] bundle done"

# 4) fusion v1/v2 对 S_HCT116 的 100 靶预测 + 评分（v1=LOCO 正确版不含 hct116 训练；v2=in-sample 标注）
$PY src/models/l15_fusion/predict_hct116_bench.py > outputs/logs/chain_fusion_predict.log 2>&1
echo "[chain] fusion predict done"

echo "[chain] ALL DONE $(date +%T)"
