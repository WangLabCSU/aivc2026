#!/bin/bash
# iter022: 顺序构建 lam70-mf14 / lam75-mf14 两个 v0.7 探针 bundle (CPU侧)
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
cd /data/resource/wanglab/aivc/aivc2026
echo "[start1] lam70mf14 $(date -u)"
python src/models/l1_signature/make_predictions.py --config configs/l1_signature/dualsrc-deep-lam70-mf14.yaml > tmp/build_lam70mf14.log 2>&1
echo "[done1] rc=$? $(date -u)"
echo "[start2] lam75mf14 $(date -u)"
python src/models/l1_signature/make_predictions.py --config configs/l1_signature/dualsrc-deep-lam75-mf14.yaml > tmp/build_lam75mf14.log 2>&1
echo "[done2] rc=$? $(date -u)"
