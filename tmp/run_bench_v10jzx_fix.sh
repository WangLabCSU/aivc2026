#!/bin/bash
# iter031 JZX 修复后重跑: local_bench 加 max_fold 逐元素帽 (镜像 build_submission)
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
cd /data/resource/wanglab/aivc/aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
echo "[bench jzx-fix] $(date -u +%H:%M:%S) start"
python src/eval/local_bench.py \
  --model-config configs/l1_signature/bench-bzr1v10-jzx-lam70-mf14.yaml \
  --contexts Q_HepG2 --tag bzr1v10jzx_Q > tmp/bench_v10jzx_Q.log 2>&1
echo "[bench jzx-fix] rc=$? $(date -u +%H:%M:%S)"
python src/eval/bench_score.py --tags bzr1v10jzx_Q --contexts Q_HepG2 > tmp/score_v10jzx.log 2>&1
echo "[score jzx-fix] rc=$? $(date -u +%H:%M:%S)"
python scripts/official_alignment.py predict --tag bzr1v10jzx_Q --family heur > tmp/pred_v10jzx.json 2>tmp/pred_v10jzx.err
echo "[pred jzx-fix] rc=$? $(date -u +%H:%M:%S)"
cat tmp/pred_v10jzx.json
echo "[chain done] $(date -u +%H:%M:%S)"
