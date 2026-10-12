#!/bin/bash
# UTC 10-13 窗口自动双发（SSL 重试版）: 等 00:00 UTC → slot1 BizardNet λ55 → 出分 → slot2 λ7.5-deep
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
cd /data/resource/wanglab/aivc/aivc2026
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc

# 等 UTC 00:00-00:09
while true; do
  H=$(date -u +%H); M=$(date -u +%M)
  if [ "$H" = "00" ] && [ "${M:0:1}" = "0" ]; then break; fi
  sleep 60
done
sleep 30

submit_retry () {  # $1=file $2=label $3=logfile → entry id
  local file=$1 label=$2 logf=$3 entry="" i
  for i in $(seq 1 24); do
    OUT=$($VCC submit "$file" -m "$label" 2>&1)
    echo "$OUT" > "$logf"
    entry=$(echo "$OUT" | grep -oE "entry [A-Za-z0-9_-]+" | head -1 | cut -d' ' -f2)
    if [ -n "$entry" ]; then echo "[OK] attempt$i entry=$entry"; echo "$entry"; return 0; fi
    if echo "$OUT" | grep -q "allowance"; then echo "[QUOTA] attempt$i"; return 1; fi
    echo "[retry] attempt$i: $(echo "$OUT" | tail -1 | head -c 90)"
    sleep 75
  done
  return 1
}

E1=$(submit_retry outputs/submissions/20261012_bizard_v10_lam55.vcc \
  "bizard-v1.0 BizardNet skip-carrier lambda=55" outputs/logs/submit_v10_lam55.log)
[ -z "$E1" ] && { echo "slot1 failed"; exit 1; }
echo "slot1 entry: $E1"

# 出分轮询（SSL 容忍: 空状态不计入失败, 最多 3h）
for i in $(seq 1 180); do
  sleep 60
  ST=$($VCC status "$E1" 2>/dev/null | grep "status:" | awk '{print $2}')
  [ -n "$ST" ] && echo "[$(date -u +%T)] poll$i: $ST"
  case "$ST" in published|failed|error) break;; esac
done
sleep 90

E2=$(submit_retry outputs/submissions/20261005_dualsrcdeep_lam75.vcc \
  "L1.8 dual-source deep lambda=7.5 (fid zero-cross probe)" outputs/logs/submit_lam75.log)
echo "slot2 entry: ${E2:-FAILED}"
echo "chain complete"
