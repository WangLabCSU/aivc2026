#!/bin/bash
# iter021 10-16 UTC 第二发: bizard-v0.6 = staged 20261005_dualsrcdeep_lam75.vcc (L1.8 dual-source deep λ7.5 fid zero-cross probe)
# 模板来源: scripts/submit_bizard_v05_1016.sh (fixed 版, 409-first + 窄 allowance 匹配); 仅改 F/L/LOG, 其余原样。
# 历史: 同 bundle 10-13 提交 (E7dtkObkTOI2olKzUEE5) 卡 launching 19h → cancel --yes, 从未计分。
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc
F=outputs/submissions/20261005_dualsrcdeep_lam75.vcc
L=bizard-v0.6
LOG=logs/submit_bizard_v06_1016.log
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
