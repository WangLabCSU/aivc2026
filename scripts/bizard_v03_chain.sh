#!/bin/bash
# Bizard v0.3 链: 表征级变异（混骨干/特征子集/行bootstrap/损失族）8 鸟。
#   worker H: 启发式鸟 Δ（bench5+submit3, 零训练骨干族）
#   worker X: 2 XGB 训练（sigdrop/rowboot）→ XGB 臂鸟 + v02_b3r 预测
#   worker M: 2 MLP 训练（mse/sigsub）→ MLP 臂鸟预测
# 之后（主控）: correlate → aggregate → bench-agg → bench-bird → A/B/C。
# 幂等: 产物存在即跳过。用法: bash scripts/bizard_v03_chain.sh [prep|all]
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
CFG=configs/bizard/v03.yaml
RUN="$PY src/models/bizard/v03_run.py --config $CFG"
LOG=logs
mkdir -p $LOG

worker_h() {
  until $RUN --build-heuristic > $LOG/bz03_heu.log 2>&1; do
    echo "[v03] heuristic retry" >&2; sleep 20; done
  echo "[v03] WORKER H DONE $(date +%T)"
}

worker_x() {
  for spec in "xgb_sigdrop xgb" "xgb_rowboot xgb"; do
    set -- $spec
    until $RUN --train --bird $1 --branch $2 > $LOG/bz03_$1_train.log 2>&1; do
      echo "[v03] train $1 retry" >&2; sleep 20; done
    echo "[v03] $1 trained $(date +%T)"
  done
  for B in xgb_s235 xgb_sigdrop xgb_rowboot v02_b3r; do
    until $RUN --predict --bird $B --submit >> $LOG/bz03_${B}_pred.log 2>&1; do
      echo "[v03] predict $B retry" >&2; sleep 20; done
    echo "[v03] $B predicted $(date +%T)"
  done
  echo "[v03] WORKER X DONE $(date +%T)"
}

worker_m() {
  for spec in "mlp_mse mlp" "mlp_sigsub mlp"; do
    set -- $spec
    until $RUN --train --bird $1 --branch $2 > $LOG/bz03_$1_train.log 2>&1; do
      echo "[v03] train $1 retry" >&2; sleep 20; done
    echo "[v03] $1 trained $(date +%T)"
  done
  for B in mlp_s235 mlp_mse mlp_sigsub; do
    until $RUN --predict --bird $B --submit >> $LOG/bz03_${B}_pred.log 2>&1; do
      echo "[v03] predict $B retry" >&2; sleep 20; done
    echo "[v03] $B predicted $(date +%T)"
  done
  echo "[v03] WORKER M DONE $(date +%T)"
}

case ${1:-all} in
  prep)
    worker_h & localH=$!
    wait $localH
    ;;
  all)
    worker_h & WH=$!
    worker_x & WX=$!
    worker_m & WM=$!
    wait $WH $WX $WM
    echo "[v03] ALL WORKERS DONE $(date +%T)"
    $RUN --correlate >> $LOG/bz03_correlate.log 2>&1
    $RUN --aggregate --submit >> $LOG/bz03_agg.log 2>&1
    echo "[v03] AGGREGATE DONE $(date +%T)"
    ;;
esac
