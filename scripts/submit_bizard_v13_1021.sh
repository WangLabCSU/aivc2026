#!/bin/bash
# iter043-16: bizard-v0.13 = HY12 + vcc2025 16官方靶注入 (轴21 官方实测, 预注册 iter043-15)
set -u
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc
F=/data/resource/wanglab/aivc/aivc2026/outputs/submissions/20261021_bizard_v13_hy12v25.vcc
L=bizard-v0.13
LOG=/data/resource/wanglab/aivc/aivc2026/logs/submit_bizard_v13_1021.log
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
      echo "[$(date -u +%H:%M:%S)] $(echo "$S" | grep status)" >> "$LOG"
    done
    echo "FINAL_STATUS:" >> "$LOG"
    "$VCC" status "$ENTRY" >> "$LOG" 2>&1
    break
  fi
  echo "$OUT" | grep -q "ALLOWANCE_EXHAUSTED\|daily" && { echo "ALLOWANCE_EXHAUSTED" >> "$LOG"; break; }
  echo "$OUT" | grep -q "409" && { echo "[409 in-progress, wait]" >> "$LOG"; sleep 75; continue; }
  sleep 75
done
echo "SUBMIT_CHAIN_DONE" >> "$LOG"
