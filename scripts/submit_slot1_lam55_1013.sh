#!/bin/bash
# UTC 10-13 晚间补发（夜链死亡后的手动恢复）: slot1 bizard v10 λ55 → 出分 → 报告
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
cd /data/resource/wanglab/aivc/aivc2026
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc

submit_retry () {  # $1=file $2=label $3=logfile → entry id (stdout)
  local file=$1 label=$2 logf=$3 entry="" i
  for i in $(seq 1 24); do
    OUT=$($VCC submit "$file" -m "$label" 2>&1)
    echo "$OUT" > "$logf"
    entry=$(echo "$OUT" | grep -oE "entry [A-Za-z0-9_-]+" | head -1 | cut -d' ' -f2)
    if [ -n "$entry" ]; then echo "[OK] attempt$i entry=$entry"; echo "$entry"; return 0; fi
    if echo "$OUT" | grep -q "allowance"; then echo "[QUOTA] attempt$i"; return 1; fi
    echo "[retry] attempt$i: $(echo "$OUT" | tail -1 | head -c 120)"
    sleep 75
  done
  return 1
}

E1=$(submit_retry outputs/submissions/20261012_bizard_v10_lam55.vcc \
  "bizard-v1.0 BizardNet skip-carrier lambda=55" logs/submit_v10_lam55.log)
[ -z "$E1" ] && { echo "slot1 FAILED after retries"; exit 1; }
echo "slot1 entry: $E1"

# 出分轮询（SSL 容忍, 最多 3h）
for i in $(seq 1 180); do
  sleep 60
  ST=$($VCC status "$E1" 2>/dev/null | grep "status:" | awk '{print $2}')
  [ -n "$ST" ] && echo "[$(date -u +%T)] poll$i: $ST"
  case "$ST" in published|failed|error) break;; esac
done
echo "[$(date -u +%T)] slot1 final status: $ST"
$VCC status "$E1" 2>&1 | tail -30
