#!/bin/bash
# iter033 v0.11 NC 官方提交链 (仅在 heur overall_pred 过门后手动启动):
#   门 = overall_pred > 0.0859 (v0.8 官方) + 余量 > 校准残差 (heur pds resid sd ~0.05)
# 构建 bizard-v0.9 artifact: build_submission → validate 内置 → vcc prep
# (不用 set -u)
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
cd /data/resource/wanglab/aivc/aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python

echo "=== [1] build submission h5ad (NC signatures, lam70 mf14) ==="
$PY src/models/l1_signature/make_predictions.py \
  --config configs/l1_signature/bzr1v11-nc-lam70-mf14.yaml \
  > tmp/build_v09nc.log 2>&1
echo "[build] rc=$? $(date -u +%H:%M:%S)"
tail -5 tmp/build_v09nc.log

echo "=== [2] vcc prep ==="
/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc prep \
  outputs/submissions/20261018_bizard_v09_nc.h5ad \
  -g data/raw/controls/gene_names.csv --perts data/raw/controls/pert_counts.csv \
  -o outputs/submissions/20261018_bizard_v09_nc.vcc \
  > tmp/prep_v09nc.log 2>&1
echo "[prep] rc=$? $(date -u +%H:%M:%S)"
tail -5 tmp/prep_v09nc.log
echo "[chain done] $(date -u +%H:%M:%S)"
