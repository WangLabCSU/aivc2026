#!/bin/bash
# iter035 幅度轴括号链 A+B: S08 (×0.8 收缩) → G115 (×1.15 微放大) 顺序 bench → score → heur predict
# 动机: sim 代理三重失真 (AMPQ75 惨败), bench-in-loop 括号幅度轴; 两支均帽内 (max|v|<3.807).
# (不用 set -u: conda deactivate 钩子 unbound variable 会误杀)
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
cd /data/resource/wanglab/aivc/aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir

for T in s08 g115; do
  T=$(echo $T)
  echo "[bench $T] $(date -u +%H:%M:%S) start"
  python src/eval/local_bench.py \
    --model-config configs/l1_signature/bench-bzr1v13-$T-lam70-mf14.yaml \
    --contexts Q_HepG2 --tag bzr1v13${T}_Q > tmp/bench_v13${T}_Q.log 2>&1
  echo "[bench $T] rc=$? $(date -u +%H:%M:%S)"
  python src/eval/bench_score.py --tags bzr1v13${T}_Q --contexts Q_HepG2 > tmp/score_v13${T}.log 2>&1
  echo "[score $T] rc=$? $(date -u +%H:%M:%S)"
done
python scripts/official_alignment.py predict --tag bzr1v13s08_Q --family heur > tmp/pred_v13s08.json 2>tmp/pred_v13s08.err
echo "[pred s08] rc=$? $(date -u +%H:%M:%S)"
python scripts/official_alignment.py predict --tag bzr1v13g115_Q --family heur > tmp/pred_v13g115.json 2>tmp/pred_v13g115.err
echo "[pred g115] rc=$? $(date -u +%H:%M:%S)"
cat tmp/pred_v13s08.json tmp/pred_v13g115.json
echo "[chain done] $(date -u +%H:%M:%S)"
