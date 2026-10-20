#!/bin/bash
# iter039 mse-axis probe: lambda micro-sweep on HY8 production family, Q_HepG2 single ctx
# Purpose: map official-scoreable u(λ) = expr_mse_unbiased_capped_norm vs pds(λ) to size
#   the mse-axis trade (never once optimized: 23 submissions all λ≥4.7 effective)
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
cd /data/resource/wanglab/aivc/aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
for L in 2.5 1.0 0.5 0.25 0.0; do
  echo "[lam $L] $(date -u +%H:%M:%S) bench start"
  python src/eval/local_bench.py --model-config tmp/mseprobe/cfg_lam${L}.yaml \
    --contexts Q_HepG2 --tag mseprobe_lam${L} > tmp/mseprobe/bench_lam${L}.log 2>&1
  echo "[lam $L] bench rc=$? $(date -u +%H:%M:%S)"
done
echo "[sweep done] $(date -u +%H:%M:%S)"
