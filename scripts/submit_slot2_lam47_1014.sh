#!/bin/bash
# iter021 slot2 (10-14 UTC 最后1发): bizard-v0.2 = λ47
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc
F=outputs/submissions/20261014_bizard_v02_lam47.vcc
L=bizard-v0.2
LOG=logs/submit_lam47_1014.log

entry=""
for i in $(seq 1 24); do
  OUT=$($VCC submit "$F" -m "$L" 2>&1)
  echo "[$(date -u +%H:%M:%S)] attempt $i:" >> "$LOG"
  echo "$OUT" >> "$LOG"
  E=$(echo "$OUT" | grep -oE "entry [A-Za-z0-9_-]+" | head -1 | cut -d' ' -f2)
  if [ -n "$E" ]; then entry=$E; break; fi
  if echo "$OUT" | grep -q "allowance"; then echo "ALLOWANCE_EXHAUSTED" >> "$LOG"; exit 1; fi
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
