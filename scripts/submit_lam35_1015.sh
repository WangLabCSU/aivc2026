#!/bin/bash
# iter021 10-15 UTC slot2: bizard-v0.4 = λ35 (构建谱系文件名 20261015_bizard_v02_lam35.vcc)
# 顺序保障: 等 slot1 (λ37) 出 ENTRY_ID 或终局失败标记后再提交, 最长等 45min 超时放行
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc
F=outputs/submissions/20261015_bizard_v02_lam35.vcc
L=bizard-v0.4
LOG=logs/submit_lam35_1015.log
WAITED=0
until grep -qE "ENTRY_ID=|SUBMIT_FAILED|ALLOWANCE_EXHAUSTED" logs/submit_lam37_1015.log 2>/dev/null; do
  sleep 30; WAITED=$((WAITED+30))
  if [ "$WAITED" -ge 2700 ]; then echo "[$(date -u +%H:%M:%S)] WAIT_SLOT1_TIMEOUT_PROCEED" >> "$LOG"; break; fi
done
echo "[$(date -u +%H:%M:%S)] slot1 gate released (waited ${WAITED}s)" >> "$LOG"
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
# poll score up to 180min
for i in $(seq 1 180); do
  sleep 60
  S=$($VCC status "$entry" 2>&1)
  echo "[$(date -u +%H:%M:%S)] $S" >> "$LOG"
  if echo "$S" | grep -qE "published|failed|error"; then break; fi
done
echo "FINAL_STATUS:" >> "$LOG"
$VCC status "$entry" >> "$LOG" 2>&1
