#!/bin/bash
# iter035-续6 RS 幅度恢复臂: bench → score → heur predict
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
cd /data/resource/wanglab/aivc/aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir

echo "[bench rs] $(date -u +%H:%M:%S) start"
python src/eval/local_bench.py \
  --model-config configs/l1_signature/bench-bzr1v15-rs-lam70-mf14.yaml \
  --contexts Q_HepG2 --tag bzr1v15rs_Q > tmp/bench_v15rs_Q.log 2>&1
echo "[bench rs] rc=$? $(date -u +%H:%M:%S)"
python src/eval/bench_score.py --tags bzr1v15rs_Q --contexts Q_HepG2 > tmp/score_v15rs.log 2>&1
echo "[score rs] rc=$? $(date -u +%H:%M:%S)"
python scripts/official_alignment.py predict --tag bzr1v15rs_Q --family heur > tmp/pred_v15rs.json 2>tmp/pred_v15rs.err
echo "[pred rs] rc=$? $(date -u +%H:%M:%S)"
cat tmp/pred_v15rs.json
echo "[chain done] $(date -u +%H:%M:%S)"
