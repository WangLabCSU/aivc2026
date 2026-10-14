#!/bin/bash
# scbasecount 修复下载：4 个未完成 h5ad + obs_metadata.parquet (GeneFull/Homo_sapiens)
# 源：GCS arc-institute-virtual-cell-atlas 2026-01-12 (与 sample_metadata file_path 一致)
# 重试：curl --retry 10 --retry-all-errors + 失败后再来一轮直到成功
set -u
B="https://storage.googleapis.com/arc-institute-virtual-cell-atlas/scbasecount/2026-01-12"
OUT="/data/resource/wanglab/aivc/aivc2026/data/public/scbasecount_download"
LOG="/data/resource/wanglab/aivc/aivc2026/logs/dl_scbasecount_repair.log"

FILES=(
  "h5ad/GeneFull/Homo_sapiens/SRX24879306.h5ad"
  "h5ad/GeneFull/Homo_sapiens/SRX22713290.h5ad"
  "h5ad/GeneFull/Homo_sapiens/SRX22713298.h5ad"
  "h5ad/GeneFull/Homo_sapiens/SRX19325980.h5ad"
  "metadata/GeneFull/Homo_sapiens/obs_metadata.parquet"
)

echo "=== start $(date -u +%FT%TZ) ===" >> "$LOG"
for f in "${FILES[@]}"; do
  base=$(basename "$f")
  expected=0
  # server-side size via HEAD
  expected=$(curl -sI "$B/$f" | grep -i "x-goog-stored-content-length" | awk '{print $2}' | tr -d '\r')
  dest="$OUT/$base"
  ok=0
  for attempt in 1 2 3 4 5 6 7 8; do
    echo "[$(date -u +%T)] attempt $attempt: $base (expect $expected bytes)" >> "$LOG"
    # 注意: ~/.curlrc 里有 --max-time 300，大文件会被 5 分钟超时打断 → 用 -q 忽略
    curl -q -sS --retry 8 --retry-all-errors --retry-delay 5 -C - --connect-timeout 30 \
      -o "$dest.part" "$B/$f" >> "$LOG" 2>&1
    actual=$(stat -c%s "$dest.part" 2>/dev/null || echo 0)
    if [ -n "$expected" ] && [ "$actual" = "$expected" ]; then
      mv "$dest.part" "$dest"
      echo "[$(date -u +%T)] OK: $base ($actual bytes)" >> "$LOG"
      ok=1
      break
    else
      echo "[$(date -u +%T)] size mismatch: got $actual want $expected, retrying" >> "$LOG"
      sleep $((attempt * 10))
    fi
  done
  [ $ok -eq 0 ] && echo "[$(date -u +%T)] FAILED after 8 attempts: $base" >> "$LOG"
done
echo "=== done $(date -u +%FT%TZ) ===" >> "$LOG"
