#!/bin/bash
# SSL 间歇干扰下的稳健提交链: slot1 → 出分 → slot2, 每步最多 20 次重试
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
cd /data/resource/wanglab/aivc/aivc2026
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc

submit_retry () {  # $1=file $2=label $3=logfile → 输出 entry id（空=失败）
  local file=$1 label=$2 logf=$3 entry="" i
  for i in $(seq 1 20); do
    OUT=$($VCC submit "$file" -m "$label" 2>&1)
    echo "$OUT" > "$logf"
    entry=$(echo "$OUT" | grep -oE "entry [A-Za-z0-9_-]+" | head -1 | cut -d' ' -f2)
    if [ -n "$entry" ]; then echo "[OK] attempt$i entry=$entry"; echo "$entry"; return 0; fi
    if echo "$OUT" | grep -q "allowance"; then echo "[QUOTA] attempt$i"; return 1; fi
    echo "[retry] attempt$i failed: $(echo "$OUT" | tail -1 | head -c 100)"
    sleep 90
  done
  return 1
}

E1=$(submit_retry outputs/submissions/20261008_bizard_v10.vcc \
  "bizard-v1.0 BizardNet skip-carrier lambda=39" outputs/logs/submit_bizard_v10_r2.log)
if [ -z "$E1" ]; then echo "slot1 failed permanently"; exit 1; fi
echo "slot1 entry: $E1"

# 等 slot1 出分（最多 3h）
for i in $(seq 1 180); do
  sleep 60
  ST=$($VCC status "$E1" 2>/dev/null | grep "status:" | awk '{print $2}')
  echo "[$(date -u +%T)] poll$i: $ST"
  case "$ST" in published|failed|error) break;; esac
done
sleep 90

E2=$(submit_retry outputs/submissions/20261012_bizard_ccrn_v02.vcc \
  "bizard-ccrn-v0.2 PBT-tuned ctx-cond MLP" outputs/logs/submit_ccrn_v02.log)
echo "slot2 entry: ${E2:-FAILED}"
echo "chain complete"
