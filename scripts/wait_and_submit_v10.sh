#!/bin/bash
# iter035-续4: 等 10-19 00:00 UTC 额度重置后提交 bizard-v0.10=HY8 (包装器, 24×75s 重试在 submit 脚本内)
while true; do
  D=$(date -u +%d%H)
  if [ "$D" -ge "1900" ] 2>/dev/null; then break; fi
  sleep 75
done
sleep 120  # 重置后缓冲 2min
bash /data/resource/wanglab/aivc/aivc2026/scripts/submit_bizard_v10_1019.sh
