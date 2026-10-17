#!/bin/bash
# iter029: bizard-v0.8 三源 bundle 构建 → vcc prep → sha256 落盘
# 链路: make_predictions.py (build h5ad) → vcc prep (h5ad→vcc) — 与 iter022 lam70mf14 同链路
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
cd /data/resource/wanglab/aivc/aivc2026
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc

echo "=== [build start] $(date -u) ==="
python src/models/l1_signature/make_predictions.py --config configs/l1_signature/dualsrc-deep-lam70-mf14-3src.yaml > tmp/build_3src.log 2>&1
echo "=== [build rc=$?] $(date -u) ==="

H5=outputs/submissions/20261017_dualsrcdeep_lam70mf14_3src.h5ad
V=outputs/submissions/20261017_dualsrcdeep_lam70mf14_3src.vcc
if [ ! -f "$H5" ]; then echo "H5_MISSING" >> tmp/build_3src.log; exit 1; fi

echo "=== [vcc prep start] $(date -u) ==="
$VCC prep "$H5" \
  -g data/raw/controls/gene_names.csv --perts data/raw/controls/pert_counts.csv \
  -o "$V" > tmp/prep_3src.log 2>&1
echo "=== [prep rc=$?] $(date -u) ==="

ls -la "$H5" "$V"
sha256sum "$V"
echo "=== [done] $(date -u) ==="
