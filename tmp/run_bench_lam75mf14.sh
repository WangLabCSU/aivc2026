#!/bin/bash
# iter022: lam75mf14 h5ad 落盘后 → Q_HepG2 bench
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
cd /data/resource/wanglab/aivc/aivc2026
H5=outputs/submissions/20261016_dualsrcdeep_lam75mf14.h5ad
for i in $(seq 1 480); do
  [ -f "$H5" ] && break
  sleep 30
done
sleep 5
echo "[bench start] $(date -u)"
python src/eval/local_bench.py --model-config configs/l1_signature/bench-dualdeep-lam75-mf14.yaml --contexts Q_HepG2 --tag dualdeep_lam75mf14_Q > tmp/bench_lam75mf14_Q.log 2>&1
echo "[bench rc=$?] $(date -u)"
