#!/bin/bash
# Bizard v0.3 bench 阶段: 等聚合 Δ 就绪 → bench-agg λ2.5 五场景 ∥ 8 鸟 Q_HepG2 bench（2 worker）。
# 用法: bash scripts/bizard_v03_bench.sh
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
RUN="$PY src/models/bizard/v03_run.py --config configs/bizard/v03.yaml"
LOG=logs
mkdir -p $LOG

for C in P_H1 Q_HepG2 R_Jurkat S_RPE1 S_HCT116; do
  while [ ! -f "data/interim/bizard/v03/agg_delta_$C.parquet" ]; do sleep 30; done
done
echo "[bench] agg deltas ready $(date +%T)"

$RUN --bench-agg --lam 2.5 > $LOG/bz03_benchagg25.log 2>&1 &
BA=$!

birdbench() {
  for spec in "$@"; do
    set -- $spec
    $RUN --bench-bird "$1" --contexts Q_HepG2 --lam "$2" > "$LOG/bz03_$1_bench.log" 2>&1
    echo "[bench] $1 done $(date +%T)"
  done
}
birdbench "heu_dual 3.5" "v02_b3r 2.5" "xgb_sigdrop 2.5" "xgb_rowboot 2.5" &
B1=$!
birdbench "xgb_s235 2.5" "mlp_s235 2.5" "mlp_sigsub 2.5" "mlp_mse 2.5" &
B2=$!

wait $BA; echo "[bench] agg λ2.5 done $(date +%T)"
wait $B1 $B2
echo "[bench] ALL BENCH DONE $(date +%T)"
