#!/bin/bash
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
cd /data/resource/wanglab/aivc/aivc2026
for d in norman k562_ess rpe1 hepg2 jurkat; do
  $PY src/data/build_signatures.py --datasets "$d" --all-perts --suffix _all > "tmp/build_sig_${d}.log" 2>&1 &
done
wait
echo ALL_DONE $(date)
