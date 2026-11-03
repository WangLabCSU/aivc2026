#!/bin/bash
# iter043-66: 10-23 UTC 00:00 额度重置后无人值守发射链 v0.16 → v0.17
set -u
cd /data/resource/wanglab/aivc/aivc2026
LOG=logs/launch_1023.log
TGT=$(date -u -d "2026-10-23 00:02:00" +%s)
echo "[$(date -u +%H:%M:%S)] timer armed, target $(date -u -d @$TGT)" >> "$LOG"
while [ "$(date +%s)" -lt "$TGT" ]; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] firing v0.16 (HY6)" >> "$LOG"
bash scripts/submit_bizard_v16_1023.sh
grep -q "SUBMIT_CHAIN_DONE" logs/submit_bizard_v16_1023.log || { echo "[warn] v16 chain log incomplete" >> "$LOG"; }
grep -q "status:   published" logs/submit_bizard_v16_1023.log && echo "[ok] v0.16 published" >> "$LOG" || echo "[FAIL] v0.16 not published" >> "$LOG"
echo "[$(date -u +%H:%M:%S)] firing v0.17 (lam62)" >> "$LOG"
bash scripts/submit_bizard_v17_1023.sh
grep -q "status:   published" logs/submit_bizard_v17_1023.log && echo "[ok] v0.17 published" >> "$LOG" || echo "[FAIL] v0.17 not published" >> "$LOG"
echo "[$(date -u +%H:%M:%S)] LAUNCH_1023_DONE" >> "$LOG"
