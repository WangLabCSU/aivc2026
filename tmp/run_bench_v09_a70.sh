#!/bin/bash
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
cd /data/resource/wanglab/aivc/aivc2026
echo "[bench a70 start] $(date -u)"
python src/eval/local_bench.py --model-config configs/l1_signature/bench-bzr1v09-a70-lam70-mf14.yaml --contexts Q_HepG2 --tag bzr1v09a70_Q > tmp/bench_v09a70_Q.log 2>&1
echo "[bench a70 rc=$?] $(date -u)"
python src/eval/bench_score.py --tags bzr1v09a70_Q --contexts Q_HepG2 > tmp/score_v09a70.log 2>&1
echo "[score a70 rc=$?] $(date -u)"
python scripts/official_alignment.py predict --tag bzr1v09a70_Q --family heur > tmp/pred_v09a70.json 2>&1
echo "[pred a70 rc=$?] $(date -u)"
cat tmp/pred_v09a70.json
