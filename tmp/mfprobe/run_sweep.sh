#!/bin/bash
# iter041: max_fold 首扫 (mf14 永冻从未在官方测过) + λ90 上段 — 本地 Q_HepG2 双臂
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
cd /data/resource/wanglab/aivc/aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
for T in mf28 mf60; do
  echo "[$T] $(date -u +%H:%M:%S) bench start"
  python src/eval/local_bench.py --model-config tmp/mfprobe/cfg_${T}.yaml \
    --contexts Q_HepG2 --tag mfprobe_${T} > tmp/mfprobe/bench_${T}.log 2>&1
  echo "[$T] bench rc=$? $(date -u +%H:%M:%S)"
done
echo "[sweep done] $(date -u +%H:%M:%S)"
