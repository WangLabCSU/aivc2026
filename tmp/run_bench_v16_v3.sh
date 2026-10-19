#!/bin/bash
# (不用 set -u: conda deactivate 钩子 unbound variable 会误杀)
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
cd /data/resource/wanglab/aivc/aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
TAG=bzr1v16
python src/eval/local_bench.py --model-config configs/l1_signature/bench-bzr1v16-v3-lam70-mf14.yaml --contexts Q_HepG2 --tag ${TAG}_Q > tmp/bench_${TAG}.log 2>&1
python src/eval/bench_score.py --tags ${TAG}_Q --contexts Q_HepG2 > tmp/score_${TAG}.log 2>&1
python scripts/official_alignment.py predict --tag ${TAG}_Q --family heur > tmp/pred_${TAG}.json 2>tmp/pred_${TAG}.err
cat tmp/pred_${TAG}.json
