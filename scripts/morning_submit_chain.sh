#!/bin/bash
# UTC 10-12 窗口自动双发: (1) bizard v10 λ39 (staged) → 等出分 → (2) CCRN v0.2 (若已构建) 或 v0.1 (staged)
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
cd /data/resource/wanglab/aivc/aivc2026
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc

# 等待 UTC 00:00 (20:00 EDT)
while [ "$(date -u +%H)" != "00" ] || [ "$(date -u +%M | cut -c1)" != "0" ]; do sleep 60; done
sleep 30

echo "[$(date -u +%T)] slot1: bizard v10"
$VCC submit outputs/submissions/20261008_bizard_v10.vcc -m "bizard-v1.0 BizardNet skip-carrier lambda=39" 2>&1
# 从日志抓 entry id 不可靠 → 用最近日志文件
# 等 slot1 完成 (published/failed 或额度错), 轮询 whoami ready
for i in $(seq 1 120); do
  sleep 60
  st=$($VCC whoami 2>&1 | grep -c "ready to submit")
  if [ "$st" = "1" ]; then break; fi
done
sleep 60

SLOT2=outputs/submissions/20261008_bizard_ccrn_v01.vcc
LABEL="bizard-ccrn-v0.1 (38d ctx-cond MLP) lambda=3.5"
if [ -f outputs/submissions/20261012_bizard_ccrn_v02.vcc ]; then
  SLOT2=outputs/submissions/20261012_bizard_ccrn_v02.vcc
  LABEL="bizard-ccrn-v0.2 PBT-tuned ctx-cond MLP"
fi
echo "[$(date -u +%T)] slot2: $SLOT2"
$VCC submit "$SLOT2" -m "$LABEL" 2>&1
echo "[$(date -u +%T)] chain complete"
