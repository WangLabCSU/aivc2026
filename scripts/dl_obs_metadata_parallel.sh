#!/bin/bash
# obs_metadata.parquet 并行分段下载器
# 单连接只有 ~15KB/s（受限于 GCS 对单流的限速），N 路并行可线性提速。
# 用 curl range 分片 → 合并 → 尺寸校验 → pyarrow 验 footer。
set -u
B="https://storage.googleapis.com/arc-institute-virtual-cell-atlas/scbasecount/2026-01-12/metadata/GeneFull/Homo_sapiens/obs_metadata.parquet"
OUT="/data/resource/wanglab/aivc/aivc2026/data/public/scbasecount_download/obs_metadata.parquet"
PART_DIR="/data/resource/wanglab/aivc/aivc2026/data/public/scbasecount_download/parts"
LOG="/data/resource/wanglab/aivc/aivc2026/logs/dl_obs_metadata_parallel.log"
TOTAL=6074238304
CHUNK=$((50*1024*1024))   # 50MB per part
NJOBS=24
mkdir -p "$PART_DIR"

N=$(( (TOTAL + CHUNK - 1) / CHUNK ))
echo "=== start $(date -u +%FT%TZ): $N parts x 50MB, $NJOBS parallel ===" >> "$LOG"

# pending list: parts not yet complete
seq 0 $((N-1)) > /tmp/obs_parts_all.txt
: > /tmp/obs_parts_todo.txt
while read i; do
  f="$PART_DIR/part_$(printf %04d $i)"
  start=$((i*CHUNK)); end=$((start+CHUNK-1))
  [ $end -ge $TOTAL ] && end=$((TOTAL-1))
  want=$((end-start+1))
  actual=$(stat -c%s "$f" 2>/dev/null || echo 0)
  [ "$actual" != "$want" ] && echo $i >> /tmp/obs_parts_todo.txt
done < /tmp/obs_parts_all.txt
todo=$(wc -l < /tmp/obs_parts_todo.txt)
echo "todo parts: $todo / $N" >> "$LOG"

download_part() {
  i=$1
  f="$PART_DIR/part_$(printf %04d $i)"
  start=$((i*CHUNK)); end=$((start+CHUNK-1))
  [ $end -ge $TOTAL ] && end=$((TOTAL-1))
  want=$((end-start+1))
  for attempt in 1 2 3 4 5 6 7 8; do
    curl -q -sS --retry 5 --retry-all-errors --retry-delay 3 --connect-timeout 30 \
      -r $start-$end -o "$f.tmp" "$B" 2>>"$LOG" && \
    actual=$(stat -c%s "$f.tmp" 2>/dev/null || echo 0) && \
    [ "$actual" = "$want" ] && { mv "$f.tmp" "$f"; return 0; }
    sleep $((attempt*5))
  done
  echo "[$(date -u +%T)] FAILED part $i" >> "$LOG"
  return 1
}
export -f download_part
export B PART_DIR LOG TOTAL CHUNK

# xargs -P parallel over todo list
cat /tmp/obs_parts_todo.txt | xargs -P $NJOBS -I{} bash -c 'download_part {}' 
rc=$?

# verify all parts
fail=0
while read i; do
  f="$PART_DIR/part_$(printf %04d $i)"
  start=$((i*CHUNK)); end=$((start+CHUNK-1))
  [ $end -ge $TOTAL ] && end=$((TOTAL-1))
  want=$((end-start+1))
  actual=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$actual" != "$want" ]; then echo "MISSING part $i" >> "$LOG"; fail=1; fi
done < /tmp/obs_parts_all.txt

if [ $fail -eq 0 ] && [ $rc -eq 0 ]; then
  echo "[$(date -u +%T)] all parts complete, concatenating..." >> "$LOG"
  cat $(seq -f "$PART_DIR/part_%04d" 0 $((N-1))) > "$OUT"
  actual=$(stat -c%s "$OUT")
  if [ "$actual" = "$TOTAL" ]; then
    echo "[$(date -u +%T)] OK: $OUT ($actual bytes)" >> "$LOG"
    rm -rf "$PART_DIR"
  else
    echo "[$(date -u +%T)] concat size mismatch: $actual vs $TOTAL" >> "$LOG"
  fi
else
  echo "[$(date -u +%T)] incomplete, rerun script to resume ($todo were pending before)" >> "$LOG"
fi
echo "=== done $(date -u +%FT%TZ) ===" >> "$LOG"
