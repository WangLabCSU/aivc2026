#!/bin/bash
# iter034 v0.12 AMPQ75 (q75选择性放大保号) 链: bench → score → heur predict
# sim5b 证据: AMP_q75_2.0 pds +0.0545 / fid 代理 dF=+0.0000 / nmae 代理 -0.0256
# (不用 set -u: conda deactivate 钩子 unbound variable 会误杀)
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
cd /data/resource/wanglab/aivc/aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir

echo "[bench ampq75] $(date -u +%H:%M:%S) start"
python src/eval/local_bench.py \
  --model-config configs/l1_signature/bench-bzr1v12-ampq75-lam70-mf14.yaml \
  --contexts Q_HepG2 --tag bzr1v12ampq75_Q > tmp/bench_v12ampq75_Q.log 2>&1
echo "[bench ampq75] rc=$? $(date -u +%H:%M:%S)"
python src/eval/bench_score.py --tags bzr1v12ampq75_Q --contexts Q_HepG2 > tmp/score_v12ampq75.log 2>&1
echo "[score ampq75] rc=$? $(date -u +%H:%M:%S)"
python scripts/official_alignment.py predict --tag bzr1v12ampq75_Q --family heur > tmp/pred_v12ampq75.json 2>tmp/pred_v12ampq75.err
echo "[pred ampq75] rc=$? $(date -u +%H:%M:%S)"
cat tmp/pred_v12ampq75.json
echo "[chain done] $(date -u +%H:%M:%S)"
