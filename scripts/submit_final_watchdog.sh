#!/bin/bash
# iter043-137: 终局统一发射链 (watchdog版, 替代逐日 submit_bizard_vNN_*.sh 裸链)
# 用法: submit_final_watchdog.sh <file.vcc> <label> <logfile.abs> <no_new_submit_after_utc "YYYY-MM-DD HH:MM">
# 依据 memory vcc-submission-ops + vcc-keyring-dbus-hang:
#   - launching 卡死 >2h 红线 -> cancel --yes 重发 (cancel 不占每日额度)
#   - 全部 vcc 调用包 timeout; DBUS_SESSION_BUS_ADDRESS 清空防 keyring 挂死
#   - 截止护栏 GUARD 之后不再发起新提交; 护栏前提交的在途 entry 一律等终态
#     (官方: 截止前受理、截止后仍在 processing 的提交有效)
#   - flock 防双实例 (at 定时器 + 手工重发并存时不双发)
#   - 成功标记 PUBLISHED_OK, 重复触发幂等退出
set -u
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc
F=$1; L=$2; LOG=$3; GUARD=${4:-"2026-11-05 21:00"}
LOCK=/data/resource/wanglab/aivc/aivc2026/tmp/.watchdog_${L}.lock
export DBUS_SESSION_BUS_ADDRESS=

guard_ts() { date -u -d "$GUARD" +%s; }

exec 9>"$LOCK"
flock -n 9 || { echo "[$(date -u +%H:%M:%S)] another instance holds lock, exit" >> "$LOG"; exit 0; }
grep -q "PUBLISHED_OK" "$LOG" 2>/dev/null && { echo "[$(date -u +%H:%M:%S)] already published earlier, idempotent exit" >> "$LOG"; exit 0; }

: > "$LOG"
echo "[$(date -u +%H:%M:%S)] watchdog start: file=$F label=$L guard=${GUARD}UTC" >> "$LOG"

for i in $(seq 1 24); do
  NOW=$(date -u +%s)
  if [ "$NOW" -gt "$(guard_ts)" ]; then
    echo "[$(date -u +%H:%M:%S)] past guard ${GUARD}UTC, stop initiating new submissions" >> "$LOG"
    break
  fi
  echo "[$(date -u +%H:%M:%S)] attempt $i:" >> "$LOG"
  OUT=$(timeout 3600 "$VCC" submit "$F" --model-name "$L" 2>&1)
  echo "$OUT" >> "$LOG"
  ENTRY=$(echo "$OUT" | grep -oE 'entry [A-Za-z0-9]+' | awk '{print $2}' | head -1)
  [ -z "$ENTRY" ] && ENTRY=$(echo "$OUT" | grep -oE 'entry: *[A-Za-z0-9]+' | awk '{print $NF}' | head -1)

  if [ -n "$ENTRY" ]; then
    echo "ENTRY_ID=$ENTRY" >> "$LOG"
    PHASE_LAST=""
    PHASE_SINCE=$(date -u +%s)
    for j in $(seq 1 300); do   # 单次在途最长 5h 观察窗
      sleep 60
      S=$(timeout 120 "$VCC" status "$ENTRY" 2>&1) || S="status_call_timeout"
      CUR=$(echo "$S" | grep -iE 'status:' | head -1 | tr -s ' ')
      [ "$CUR" != "$PHASE_LAST" ] && { echo "[$(date -u +%H:%M:%S)] $CUR" >> "$LOG"; PHASE_LAST="$CUR"; PHASE_SINCE=$(date -u +%s); }
      if echo "$S" | grep -qiE "status: *published"; then
        echo "[$(date -u +%H:%M:%S)] published" >> "$LOG"; PUBLISHED=1; break 2
      fi
      if echo "$S" | grep -qiE "status: *(failed|error|cancelled)"; then
        echo "[$(date -u +%H:%M:%S)] terminal failure -> refire" >> "$LOG"; break
      fi
      ELAPSED=$(( $(date -u +%s) - PHASE_SINCE ))
      if [ "$ELAPSED" -gt 7200 ] && [ "$(date -u +%s)" -lt "$(guard_ts)" ]; then
        # 2h 红线: 同一 phase 停滞超 2h 且未过护栏 -> cancel 重发; 护栏后只等不动
        echo "[$(date -u +%H:%M:%S)] phase stuck ${ELAPSED}s > 2h red line -> cancel + refire" >> "$LOG"
        timeout 120 "$VCC" cancel "$ENTRY" --yes >> "$LOG" 2>&1
        break
      fi
    done
    [ "${PUBLISHED:-0}" -eq 1 ] && break
    continue
  fi

  echo "$OUT" | grep -qiE "ALLOWANCE_EXHAUSTED|daily submission allowance" && { echo "ALLOWANCE_EXHAUSTED_TODAY" >> "$LOG"; break; }
  echo "$OUT" | grep -q "409" && { echo "[$(date -u +%H:%M:%S)] 409 in-progress slot busy, wait" >> "$LOG"; sleep 75; continue; }
  sleep 75
done

echo "FINAL_STATUS:" >> "$LOG"
[ -n "${ENTRY:-}" ] && timeout 120 "$VCC" status "$ENTRY" >> "$LOG" 2>&1
if [ "${PUBLISHED:-0}" -eq 1 ]; then
  echo "PUBLISHED_OK" >> "$LOG"
else
  echo "NOT_PUBLISHED_CHECK_LOG" >> "$LOG"
fi
echo "SUBMIT_CHAIN_DONE" >> "$LOG"
