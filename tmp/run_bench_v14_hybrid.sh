#!/bin/bash
# iter035-续2: NC+yh 混合臂 (HY/HYf) bench 链 — build 完成后启动
# (不用 set -u: conda deactivate 钩子 unbound variable 会误杀)
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
cd /data/resource/wanglab/aivc/aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir

for T in hy hyf; do
  echo "[bench $T] $(date -u +%H:%M:%S) start"
  python src/eval/local_bench.py \
    --model-config configs/l1_signature/bench-bzr1v14-$T-lam70-mf14.yaml \
    --contexts Q_HepG2 --tag bzr1v14${T}_Q > tmp/bench_v14${T}_Q.log 2>&1
  echo "[bench $T] rc=$? $(date -u +%H:%M:%S)"
  python src/eval/bench_score.py --tags bzr1v14${T}_Q --contexts Q_HepG2 > tmp/score_v14${T}.log 2>&1
  echo "[score $T] rc=$? $(date -u +%H:%M:%S)"
done
python scripts/official_alignment.py predict --tag bzr1v14hy_Q --family heur > tmp/pred_v14hy.json 2>tmp/pred_v14hy.err
echo "[pred hy] rc=$? $(date -u +%H:%M:%S)"
python scripts/official_alignment.py predict --tag bzr1v14hyf_Q --family heur > tmp/pred_v14hyf.json 2>tmp/pred_v14hyf.err
echo "[pred hyf] rc=$? $(date -u +%H:%M:%S)"
cat tmp/pred_v14hy.json tmp/pred_v14hyf.json
echo "[chain done] $(date -u +%H:%M:%S)"
