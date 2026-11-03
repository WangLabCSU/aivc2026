#!/bin/bash
# iter043-135: 10-25 第2发 — HY8×λ70 复刻 (v0.18 bundle 复用) as bizard-v0.21
# 目的: 网格 heur 最优 (0.0915) 升级为当前 last; 消耗当日作废额度; 链路保温
# 注: 只最后一发计奖 — v0.21 期望 0.0915 ≥ v0.20 hykens 0.0914, 零下行
set -u
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc
F=/data/resource/wanglab/aivc/aivc2026/outputs/submissions/20261024_bizard_v18_final_hy8_nogeneric.prep.vcc
L=bizard-v0.21
LOG=/data/resource/wanglab/aivc/aivc2026/logs/submit_bizard_v21_1025.log
export DBUS_SESSION_BUS_ADDRESS=  # keyring 挂死防护 (memory vcc-keyring-dbus-hang)

: > "$LOG"
for i in $(seq 1 24); do
  echo "[$(date +%H:%M:%S)] attempt $i:" >> "$LOG"
  OUT=$(timeout 3600 "$VCC" submit "$F" --model-name "$L" 2>&1)
  echo "$OUT" >> "$LOG"
  ENTRY=$(echo "$OUT" | grep -oE 'entry [A-Za-z0-9]+' | awk '{print $2}' | head -1)
  if [ -n "$ENTRY" ]; then
    echo "ENTRY_ID=$ENTRY" >> "$LOG"
    for j in $(seq 1 180); do
      sleep 60
      S=$("$VCC" status "$ENTRY" 2>&1)
      echo "$S" | grep -qE "published|failed|error" && break
      echo "[$(date -u +%H:%M:%S)] $(echo "$S" | grep -i status)" >> "$LOG"
    done
    echo "FINAL_STATUS:" >> "$LOG"
    "$VCC" status "$ENTRY" >> "$LOG" 2>&1
    break
  fi
  echo "$OUT" | grep -qiE "ALLOWANCE_EXHAUSTED|daily" && { echo "ALLOWANCE_EXHAUSTED" >> "$LOG"; break; }
  echo "$OUT" | grep -q "409" && { echo "[409 in-progress, wait]" >> "$LOG"; sleep 75; continue; }
  sleep 75
done
echo "SUBMIT_CHAIN_DONE" >> "$LOG"
