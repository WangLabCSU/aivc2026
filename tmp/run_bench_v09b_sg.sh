#!/bin/bash
# iter030-修正6 v0.9b sign-gate SG100/SG50 双链: bench → score → heur predict
set -u
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
cd /data/resource/wanglab/aivc/aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir

for TAG in sg100 sg50; do
  echo "[bench $TAG] $(date -u +%H:%M:%S) start"
  python src/eval/local_bench.py \
    --model-config configs/l1_signature/bench-bzr1v09b-${TAG}-lam70-mf14.yaml \
    --contexts Q_HepG2 --tag bzr1v09b${TAG}_Q > tmp/bench_v09b${TAG}_Q.log 2>&1
  echo "[bench $TAG] rc=$? $(date -u +%H:%M:%S)"
  python src/eval/bench_score.py --tags bzr1v09b${TAG}_Q --contexts Q_HepG2 > tmp/score_v09b${TAG}.log 2>&1
  echo "[score $TAG] rc=$? $(date -u +%H:%M:%S)"
  python scripts/official_alignment.py predict --tag bzr1v09b${TAG}_Q --family heur > tmp/pred_v09b${TAG}.json 2>tmp/pred_v09b${TAG}.err
  echo "[pred $TAG] rc=$? $(date -u +%H:%M:%S)"
  cat tmp/pred_v09b${TAG}.json
done
echo "[chain done] $(date -u +%H:%M:%S)"
