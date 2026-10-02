#!/bin/bash
# 夜间任务: plate12 剩余区间并行下载（前置 wget 已停于 C0 字节，本脚本只取 [C0, T)）。
# 分块 256MiB × 5 并发，逐块校验大小，全部通过后按序拼接并 mv 为 plate12.h5ad。
set -u
ROOT=/data/resource/wanglab/aivc/aivc2026
D=$ROOT/data/public/tahoe
PART=$D/plate12.h5ad.part
PARTS=$D/p12_parts
URL="https://storage.googleapis.com/arc-institute-virtual-cell-atlas/tahoe100M/2025-02-25/h5ad/plate12_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
T=37495736876
C0=$(stat -c%s "$PART")
CHUNK=268435456
mkdir -p "$PARTS"

# 生成计划: name|start|end|expected
PLAN=$PARTS/plan.txt
python3 - "$C0" "$T" "$CHUNK" "$PARTS" > "$PLAN" <<'PY'
import sys
c0, T, chunk, parts = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
i = 0; s = c0
while s < T:
    e = min(s + chunk - 1, T - 1)
    print(f"{parts}/blk_{i:04d}|{s}|{e}|{e-s+1}")
    s = e + 1; i += 1
PY
echo "chunks: $(wc -l < "$PLAN"), remaining $((T-C0)) bytes"

fetch_one() {
    line="$1"
    name=$(echo "$line" | cut -d'|' -f1)
    s=$(echo "$line" | cut -d'|' -f2)
    e=$(echo "$line" | cut -d'|' -f3)
    exp=$(echo "$line" | cut -d'|' -f4)
    [ -f "$name" ] && [ "$(stat -c%s "$name" 2>/dev/null || echo 0)" = "$exp" ] && return 0
    for att in 1 2 3 4 5; do
        curl -s --retry 3 --retry-all-errors --max-time 600 \
            --speed-limit 1048576 --speed-time 30 \
            -r "$s-$e" -o "$name.tmp" "$URL" \
            && [ "$(stat -c%s "$name.tmp" 2>/dev/null || echo 0)" = "$exp" ] \
            && mv "$name.tmp" "$name" && return 0
        sleep $((att*10))
    done
    echo "FAIL $name" >&2
    return 1
}
export -f fetch_one
export URL

for pass in 1 2 3; do
    cat "$PLAN" | xargs -P 12 -I{} bash -c 'fetch_one "$@"' _ {}
    n_miss=0
    while IFS='|' read -r name s e exp; do
        [ -f "$name" ] && [ "$(stat -c%s "$name")" = "$exp" ] || n_miss=$((n_miss+1))
    done < "$PLAN"
    echo "pass $pass: missing=$n_miss"
    [ "$n_miss" = 0 ] && break
done

# 校验全部块
n_bad=0
while IFS='|' read -r name s e exp; do
    [ -f "$name" ] && [ "$(stat -c%s "$name")" = "$exp" ] || { n_bad=$((n_bad+1)); echo "BAD $name"; }
done < "$PLAN"
if [ "$n_bad" != 0 ]; then echo "ABORT: $n_bad bad chunks"; exit 1; fi

# 拼接（追加到 .part 后原子改名）
echo "assembling..."
cat $(cut -d'|' -f1 "$PLAN" | sort) >> "$PART"
FS=$(stat -c%s "$PART")
if [ "$FS" != "$T" ]; then echo "ABORT: final size $FS != $T"; exit 1; fi
mv "$PART" "$D/plate12.h5ad"
rm -rf "$PARTS"
echo "PLATE12_DONE size=$FS $(date)" >> "$ROOT/outputs/logs/dl_plate12.log"
echo "ALL_DONE size=$FS"
