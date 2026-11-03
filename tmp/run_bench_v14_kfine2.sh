#!/bin/bash
# iter035-续3: HY3/HY8 full-pipeline bench (parquet 已建) — 对照 k 曲线
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
cd /data/resource/wanglab/aivc/aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir

for T in hy3 hy8; do
  echo "[bench $T] $(date -u +%H:%M:%S) start"
  python src/eval/local_bench.py \
    --model-config configs/l1_signature/bench-bzr1v14-$T-lam70-mf14.yaml \
    --contexts Q_HepG2 --tag bzr1v14${T}_Q > tmp/bench_v14${T}_Q.log 2>&1
  echo "[bench $T] rc=$? $(date -u +%H:%M:%S)"
  python src/eval/bench_score.py --tags bzr1v14${T}_Q --contexts Q_HepG2 > tmp/score_v14${T}.log 2>&1
  echo "[score $T] rc=$? $(date -u +%H:%M:%S)"
done
python scripts/official_alignment.py predict --tag bzr1v14hy3_Q --family heur > tmp/pred_v14hy3.json 2>tmp/pred_v14hy3.err
python scripts/official_alignment.py predict --tag bzr1v14hy8_Q --family heur > tmp/pred_v14hy8.json 2>tmp/pred_v14hy8.err
echo "[pred heur] rc=$? $(date -u +%H:%M:%S)"
python scripts/official_alignment.py predict --tag bzr1v14hy3_Q --family official-mode > tmp/pred_v14hy3_om.json 2>tmp/pred_v14hy3om.err
python scripts/official_alignment.py predict --tag bzr1v14hy8_Q --family official-mode > tmp/pred_v14hy8_om.json 2>tmp/pred_v14hy8om.err
echo "[pred om] rc=$? $(date -u +%H:%M:%S)"
cat tmp/pred_v14hy3.json tmp/pred_v14hy8.json
echo "[chain2 done] $(date -u +%H:%M:%S)"
