#!/bin/bash
# iter027: gamma sharpening 双臂验证闭环 — cell-eval2 eval → bench_score → official predict
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
cd /data/resource/wanglab/aivc/aivc2026
REF=data/interim/local_bench/Q_HepG2/ref.h5ad
for G in 0.5 1.5; do
  TAG="sharpen_g${G}"
  PRED="outputs/predictions/${TAG}/Q_HepG2_pred.h5ad"
  OUT="outputs/metrics/local_bench/${TAG}/Q_HepG2_eval"
  echo "=== [eval ${TAG}] $(date -u) ==="
  python -c "
import anndata as ad, scipy.sparse as sp
a = ad.read_h5ad('${PRED}')
a.var_names = [str(x) for x in a.var_names]
a.var.index.name = None
X = sp.csr_matrix(a.X)
X.eliminate_zeros()
a.X = X
a.write_h5ad('/data/resource/wanglab/aivc/tmpdir/${TAG}_pred_fixed.h5ad', compression='gzip')
print('var fixed + zeros eliminated', a.shape, X.nnz)
"
  cell-eval2 run -ap "/data/resource/wanglab/aivc/tmpdir/${TAG}_pred_fixed.h5ad" -ar "$REF" \
    --preset vcc2026 --pert-col target_gene --control non-targeting \
    --input-type counts --set de.backend=pdex -o "$OUT"
  echo "=== [eval ${TAG} rc=$?] $(date -u) ==="
done
echo "=== [bench_score] $(date -u) ==="
python src/eval/bench_score.py --tags sharpen_g0.5,sharpen_g1.5 --contexts Q_HepG2 \
  --out outputs/metrics/local_bench/anchor_normalized_sharpen.json
echo "=== [official predict] $(date -u) ==="
for G in 0.5 1.5; do
  echo "--- sharpen_g${G} ---"
  python scripts/official_alignment.py predict --tag "sharpen_g${G}" --family heur
done
echo "=== [done] $(date -u) ==="
