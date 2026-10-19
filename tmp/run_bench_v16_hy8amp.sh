#!/bin/bash
# iter037-续6: bzr1v16 = HY8 + amp_max_5src (100 bench 靶 sel 位 max(|yc|, A_sq_5src))
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
cd /data/resource/wanglab/aivc/aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir

TAG=bzr1v16hy8amp
CFG=configs/l1_signature/bench-bzr1v16-hy8amp-lam70-mf14.yaml

python src/eval/local_bench.py --model-config "$CFG" --contexts Q_HepG2 --tag ${TAG}_Q \
  > tmp/bench_${TAG}_Q.log 2>&1
python src/eval/bench_score.py --tags ${TAG}_Q --contexts Q_HepG2 \
  > tmp/score_${TAG}.log 2>&1
python scripts/official_alignment.py predict --tag ${TAG}_Q --family heur \
  > tmp/pred_${TAG}.json 2>tmp/pred_${TAG}.err
echo "=== heur overall_pred ==="
cat tmp/pred_${TAG}.json
