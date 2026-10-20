#!/bin/bash
# iter039 10-20 UTC 发1: bizard-v0.11 = λ2.5 mse轴官方探针 (HY8 族)
# 动机: 23发官方 mse 恒 0 (u≥1); λ微扫显示本地 u(λ2.5)=0.9469→mse_score 0.053。
# INFO shot: 采样官方 u(λ) 曲线下段, 校准未来所有 λ 决策。
# 模板来源: scripts/submit_bizard_v09_1018.sh; 仅改 F/L/LOG。
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc
F=outputs/submissions/20261020_bizard_v11_lam25.vcc
L=bizard-v0.11
LOG=logs/submit_bizard_v11_1020.log
entry=""
for i in $(seq 1 24); do
  OUT=$($VCC submit "$F" -m "$L" 2>&1)
  echo "[$(date -u +%H:%M:%S)] attempt $i:" >> "$LOG"
  echo "$OUT" >> "$LOG"
  E=$(echo "$OUT" | grep -oE "entry [A-Za-z0-9_-]+" | head -1 | cut -d' ' -f2)
  if [ -n "$E" ]; then entry=$E; break; fi
  if echo "$OUT" | grep -q "already has a submission in progress"; then
    echo "[$(date -u +%H:%M:%S)] 409 one-at-a-time, keep waiting" >> "$LOG"
  elif echo "$OUT" | grep -qiE "allowance.*(exhaust|exceed|used|no submissions)|no .*allowance"; then
    echo "ALLOWANCE_EXHAUSTED" >> "$LOG"; exit 1
  fi
  sleep 75
done
if [ -z "$entry" ]; then echo "SUBMIT_FAILED" >> "$LOG"; exit 1; fi
echo "ENTRY_ID=$entry" >> "$LOG"
echo "$entry"
for i in $(seq 1 180); do
  sleep 60
  S=$($VCC status "$entry" 2>&1)
  echo "[$(date -u +%H:%M:%S)] $S" >> "$LOG"
  if echo "$S" | grep -qE "published|failed|error"; then break; fi
done
echo "FINAL_STATUS:" >> "$LOG"
$VCC status "$entry" >> "$LOG" 2>&1
