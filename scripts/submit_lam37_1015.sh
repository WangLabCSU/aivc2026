#!/bin/bash
# iter021 10-15 UTC slot1: bizard-v0.3 = λ37 (构建谱系文件名 20261015_bizard_v02_lam37.vcc)
# ⚠️ DEPRECATED TEMPLATE — 本脚本的 allowance 判定是旧版宽匹配 (L16 grep -q "allowance")。
# 已知坑: HTTP 409 "already has a submission in progress" 文案含 "allowance" 一词,
# 会被 L16 误判为 ALLOWANCE_EXHAUSTED 而提前退出 (slot2 假警报实录, 10-15)。
# 修复版模板见 scripts/submit_lam35_1015.sh (先判 409 再窄匹配 allowance)。
# 本文件仅作 10-15 实际运行的历史留档, 禁止作为新提交脚本的复制源。
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc
F=outputs/submissions/20261015_bizard_v02_lam37.vcc
L=bizard-v0.3
LOG=logs/submit_lam37_1015.log
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
