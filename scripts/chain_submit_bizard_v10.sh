#!/bin/bash
# 等 ensemble 出分后自动提交 bizard v10
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
cd /data/resource/wanglab/aivc/aivc2026
for i in $(seq 1 90); do
  st=$(vcc status vxpevKCTFdjIs9Smp0ul 2>/dev/null | grep "status:")
  echo "[$(date +%H:%M:%S)] poll$i: $st"
  if echo "$st" | grep -qE "published|scored|failed"; then break; fi
  sleep 60
done
sleep 10
vcc submit outputs/submissions/20261008_bizard_v10.vcc -m "bizard-v1.0 BizardNet skip-carrier lambda=39" 2>&1
