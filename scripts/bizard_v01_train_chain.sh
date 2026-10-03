#!/bin/bash
# Bizard v0.1 鸟群训练链: b1..b5 顺序, 每鸟 xgb∥mlp（b6_g 复用 v35 免训）。
# 完成判据: .train_done 或 (.xgb_done 且 .mlp_done); 复用鸟视为完成。
# 幂等: 可中断续跑（进程存活检测 pgrep, 防双启动）。
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
W=data/interim/bizard/v01

bird_done() {
  local b=$1
  [ -f "$W/$b/.train_done" ] && return 0
  [ -f "$W/$b/.xgb_done" ] && [ -f "$W/$b/.mlp_done" ] && return 0
  return 1
}

for B in b1_p b2_q b3_r b4_s1 b5_s2; do
  if bird_done "$B"; then echo "[bzchain] $B already done"; continue; fi
  while ! bird_done "$B"; do
    if ! pgrep -f "train_bird.py --bird $B" > /dev/null; then
      echo "[bzchain] $B start $(date +%T)"
      $PY src/models/bizard/train_bird.py --bird $B --branch xgb > logs/bz01_${B}_xgb.log 2>&1 &
      PX=$!
      $PY src/models/bizard/train_bird.py --bird $B --branch mlp > logs/bz01_${B}_mlp.log 2>&1 &
      PM=$!
      wait $PX $PM
      bird_done "$B" || { echo "[bzchain] $B FAILED, retry in 60s" >&2; sleep 60; }
    else
      sleep 20
    fi
  done
  echo "[bzchain] $B done $(date +%T)"
done
echo "[bzchain] ALL BIRDS DONE $(date +%T)"
