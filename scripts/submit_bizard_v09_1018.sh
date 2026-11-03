#!/bin/bash
# iter035-续2 10-18 UTC: bizard-v0.9 = 20261018_bizard_v09_hy12.vcc (NC+yh 混合臂 HY12, heur 0.0905/anchor 0.3144 全场最佳, 门0.0859)
# 模板来源: scripts/submit_bizard_v08_1017.sh; 仅改 F/L/LOG, 其余原样。
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc
F=outputs/submissions/20261018_bizard_v09_hy12.vcc
L=bizard-v0.9
LOG=logs/submit_bizard_v09_1018.log
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
# poll score up to 180min; 卡 launching 2h → 人工 cancel --yes 规则 (vcc-submission-ops)
for i in $(seq 1 180); do
  sleep 60
  S=$($VCC status "$entry" 2>&1)
  echo "[$(date -u +%H:%M:%S)] $S" >> "$LOG"
  if echo "$S" | grep -qE "published|failed|error"; then break; fi
done
echo "FINAL_STATUS:" >> "$LOG"
$VCC status "$entry" >> "$LOG" 2>&1
