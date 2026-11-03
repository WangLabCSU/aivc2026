#!/bin/bash
# iter终局B: 11-05 01:02 UTC 闭合弹定时器 — v0.23 = HY8 x lam70 复刻 (最后一发计奖)
# nohup 脱离 session; 循环每 60s 检查, 到点触发 submit 链
# 缓冲分析: 01:02 发射 -> 上传+评分 ~0.5-3h -> published ~04:00 UTC;
#           若卡死 2h 红线 03:02 由 loop 心跳 cancel 重发, 距 23:59 截止仍有 ~21h
set -u
cd /data/resource/wanglab/aivc/aivc2026
LOG=logs/launch_1105.log
TGT=$(date -u -d "2026-11-05 01:02:00" +%s)
echo "[$(date -u +%H:%M:%S)] closing-shot timer armed, target $(date -u -d @$TGT)" >> "$LOG"
while [ "$(date +%s)" -lt "$TGT" ]; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] firing closing shot bizard-v0.23 (HY8 lam70)" >> "$LOG"
bash scripts/submit_bizard_v23_1105.sh
grep -q "SUBMIT_CHAIN_DONE" logs/submit_bizard_v23_1105.log || echo "[warn] v23 chain log incomplete" >> "$LOG"
grep -q "status:   published" logs/submit_bizard_v23_1105.log && echo "[ok] v0.23 published - FINAL ENTRY SEALED" >> "$LOG" || echo "[FAIL] v0.23 not published - loop heartbeat must intervene cancel+refire" >> "$LOG"
echo "[$(date -u +%H:%M:%S)] LAUNCH_1105_DONE" >> "$LOG"
