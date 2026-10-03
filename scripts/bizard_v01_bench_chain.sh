#!/bin/bash
# Bizard v0.1 每鸟收尾链: 鸟训练完成 → bench 场景预测 → A/B/C 预测 → bench 评分（主场景+Q）。
# 幂等: pred parquet / bizard_summary.json 存在即跳过。
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
W=data/interim/bizard/v01

declare -A MAIN=( [b1_p]=P_H1 [b2_q]=Q_HepG2 [b3_r]=R_Jurkat [b4_s1]=S_RPE1 [b5_s2]=S_HCT116 )

bird_done() {
  local b=$1
  [ -f "$W/$b/.train_done" ] && return 0
  [ -f "$W/$b/.xgb_done" ] && [ -f "$W/$b/.mlp_done" ] && return 0
  return 1
}

for B in b2_q b3_r b4_s1 b5_s2; do
  while ! bird_done "$B"; do sleep 30; done
  echo "[bzbench] $B ready $(date +%T)"
  CTX="${MAIN[$B]},Q_HepG2"
  CTX=$(echo "$CTX" | tr ',' '\n' | awk '!seen[$0]++' | paste -sd, -)
  until $PY src/models/bizard/predict.py --bird $B >> logs/bz01_pred_${B}.log 2>&1; do
    echo "[bzbench] $B predict failed, retry 30s" >&2; sleep 30
  done
  until $PY src/models/bizard/predict.py --bird $B --submit >> logs/bz01_pred_${B}.log 2>&1; do
    echo "[bzbench] $B submit-predict failed, retry 30s" >&2; sleep 30
  done
  until $PY src/models/bizard/bench_bird.py --tag bz01_$B --lam 2.5 --contexts "$CTX" \
      >> logs/bz01_bench_${B}.log 2>&1; do
    echo "[bzbench] $B bench failed, retry 60s" >&2; sleep 60
  done
  echo "[bzbench] $B benched $(date +%T)"
done
echo "[bzbench] ALL BENCH DONE $(date +%T)"
