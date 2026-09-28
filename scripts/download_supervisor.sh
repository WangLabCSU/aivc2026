#!/bin/bash
# 下载监督循环：等待当前批次退出后重跑（wget -c 断点续传），直到全部 .done
# 用法: nohup bash scripts/download_supervisor.sh > outputs/logs/download_supervisor.log 2>&1 &
set -u
ROOT=/data/resource/wanglab/aivc/aivc2026
cd "$ROOT"
round=0
while true; do
  round=$((round+1))
  gcs_done=$(ls data/public/vcc2025/*.done 2>/dev/null | wc -l)
  zen_done=$(ls data/public/scperturb/*.done 2>/dev/null | wc -l)
  echo "[$(date +%F\ %T)] round=$round done: gcs=$gcs_done/7 zenodo=$zen_done/7"
  [ "$gcs_done" -ge 7 ] && [ "$zen_done" -ge 7 ] && { echo ALL COMPLETE; break; }
  pgrep -f 'download_pool.sh (gcs|zenodo)' > /dev/null || {
    [ "$gcs_done" -lt 7 ] && nohup bash scripts/download_pool.sh gcs   >> outputs/logs/download_gcs.log    2>&1 &
    [ "$zen_done" -lt 7 ] && nohup bash scripts/download_pool.sh zenodo >> outputs/logs/download_zenodo.log 2>&1 &
  }
  sleep 300
done
