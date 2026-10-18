#!/bin/bash
# iter031 v0.10 跨pert对比归一 (HCX/JZX) 双链: bench → score → heur predict
# sim3 证据: H_center_x +0.0454 / J_zscore_x +0.0427 (S1/S2 PASS, sim≈bench raw 0.8130/0.8156)
# (不用 set -u: conda deactivate 钩子 unbound variable 会误杀)
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
cd /data/resource/wanglab/aivc/aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir

for TAG in hcx jzx; do
  echo "[bench $TAG] $(date -u +%H:%M:%S) start"
  python src/eval/local_bench.py \
    --model-config configs/l1_signature/bench-bzr1v10-${TAG}-lam70-mf14.yaml \
    --contexts Q_HepG2 --tag bzr1v10${TAG}_Q > tmp/bench_v10${TAG}_Q.log 2>&1
  echo "[bench $TAG] rc=$? $(date -u +%H:%M:%S)"
  python src/eval/bench_score.py --tags bzr1v10${TAG}_Q --contexts Q_HepG2 > tmp/score_v10${TAG}.log 2>&1
  echo "[score $TAG] rc=$? $(date -u +%H:%M:%S)"
  python scripts/official_alignment.py predict --tag bzr1v10${TAG}_Q --family heur > tmp/pred_v10${TAG}.json 2>tmp/pred_v10${TAG}.err
  echo "[pred $TAG] rc=$? $(date -u +%H:%M:%S)"
  cat tmp/pred_v10${TAG}.json
done
echo "[chain done] $(date -u +%H:%M:%S)"
