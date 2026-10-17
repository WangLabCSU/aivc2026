#!/bin/bash
# iter029: 第三源加源双臂 A/B — k562_ess_all (R1) / jurkat_all+rpe1_all (R2) vs anchor λ70mf14
# 闭环: local_bench (config-only 加源) → bench_score → official_alignment predict --family heur
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
cd /data/resource/wanglab/aivc/aivc2026

echo "=== [R1 bench start] $(date -u) ==="
python src/eval/local_bench.py --model-config configs/l1_signature/bench-dualdeep3src-k562.yaml \
  --contexts Q_HepG2 --tag dualdeep3src_k562_lam70mf14_Q > tmp/bench_3src_R1.log 2>&1
echo "=== [R1 rc=$?] $(date -u) ==="

echo "=== [R2 bench start] $(date -u) ==="
python src/eval/local_bench.py --model-config configs/l1_signature/bench-dualdeep4src-jr.yaml \
  --contexts Q_HepG2 --tag dualdeep4src_jr_lam70mf14_Q > tmp/bench_3src_R2.log 2>&1
echo "=== [R2 rc=$?] $(date -u) ==="

echo "=== [bench_score] $(date -u) ==="
python src/eval/bench_score.py --tags dualdeep3src_k562_lam70mf14_Q,dualdeep4src_jr_lam70mf14_Q --contexts Q_HepG2 \
  --out outputs/metrics/local_bench/anchor_normalized_3src.json

echo "=== [official predict] $(date -u) ==="
python scripts/official_alignment.py predict --tag "dualdeep3src_k562_lam70mf14_Q" --family heur
python scripts/official_alignment.py predict --tag "dualdeep4src_jr_lam70mf14_Q" --family heur
echo "=== [done] $(date -u) ==="
