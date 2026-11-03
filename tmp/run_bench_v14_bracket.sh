#!/bin/bash
# iter035-续2: HY50 / HY12 DE 括号 bench 链 (k=25 HY 已跑: 0.0887)
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
cd /data/resource/wanglab/aivc/aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir

for T in hy50 hy12; do
  echo "[bench $T] $(date -u +%H:%M:%S) start"
  python src/eval/local_bench.py \
    --model-config configs/l1_signature/bench-bzr1v14-$T-lam70-mf14.yaml \
    --contexts Q_HepG2 --tag bzr1v14${T}_Q > tmp/bench_v14${T}_Q.log 2>&1
  echo "[bench $T] rc=$? $(date -u +%H:%M:%S)"
  python src/eval/bench_score.py --tags bzr1v14${T}_Q --contexts Q_HepG2 > tmp/score_v14${T}.log 2>&1
  echo "[score $T] rc=$? $(date -u +%H:%M:%S)"
done
python scripts/official_alignment.py predict --tag bzr1v14hy50_Q --family heur > tmp/pred_v14hy50.json 2>tmp/pred_v14hy50.err
echo "[pred hy50] rc=$? $(date -u +%H:%M:%S)"
python scripts/official_alignment.py predict --tag bzr1v14hy12_Q --family heur > tmp/pred_v14hy12.json 2>tmp/pred_v14hy12.err
echo "[pred hy12] rc=$? $(date -u +%H:%M:%S)"
cat tmp/pred_v14hy50.json tmp/pred_v14hy12.json
echo "[chain done] $(date -u +%H:%M:%S)"
