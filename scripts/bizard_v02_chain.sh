#!/bin/bash
# Bizard v0.2 链: 阶段1 = 6 鸟训练(2 worker, 每鸟 xgb∥mlp) + 2 bench worker 并行收尾;
# 阶段2 = PBT 代际(v02_pbt.py 选头鸟/末位鸟) → 2 只变异鸟重训+重评 → 聚合。
# 幂等: 各步骤产物存在即跳过。
# 用法: bash scripts/bizard_v02_chain.sh [gen1|gen2|all]
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
CFG=configs/bizard/v02.yaml
W=data/interim/bizard/v02
LOG=logs
mkdir -p $LOG
PHASE=${1:-all}
ROOT_OUT=outputs/metrics/local_bench

bird_done() { [ -f "$W/$1/.train_done" ]; }

train_bird_full() {
  local B=$1
  bird_done "$B" && { echo "[v02] $B train done"; return 0; }
  $PY src/models/bizard/v02_run.py --config $CFG --train --bird $B --branch xgb \
      > $LOG/bz02_${B}_xgb.log 2>&1 &
  local PX=$!
  $PY src/models/bizard/v02_run.py --config $CFG --train --bird $B --branch mlp \
      > $LOG/bz02_${B}_mlp.log 2>&1 &
  local PM=$!
  wait $PX $PM
  bird_done "$B" && echo "[v02] $B trained $(date +%T)" || { echo "[v02] $B TRAIN FAILED" >&2; return 1; }
}

finish_bird() {
  local B=$1
  while ! bird_done "$B"; do sleep 20; done
  if [ -f "$ROOT_OUT/bz02_$B/bizard_summary.json" ]; then echo "[v02] $B already benched"; return 0; fi
  until $PY src/models/bizard/v02_run.py --config $CFG --predict --bird $B \
      >> $LOG/bz02_${B}_pred.log 2>&1; do echo "[v02] $B predict retry" >&2; sleep 20; done
  until $PY src/models/bizard/v02_run.py --config $CFG --bench --bird $B \
      >> $LOG/bz02_${B}_bench.log 2>&1; do echo "[v02] $B bench retry" >&2; sleep 30; done
  echo "[v02] $B benched $(date +%T)"
}

gen1() {
  # 训练 worker A/B + bench worker A/B 四进程并行
  (
    for B in b1_p b2_q b3_r; do train_bird_full "$B"; done
    for B in b1_p b2_q b3_r; do finish_bird "$B"; done
    echo "[v02] WORKER A ALL DONE $(date +%T)"
  ) > $LOG/bz02_workerA.log 2>&1 &
  local WA=$!
  (
    for B in b4_s1 b5_s2 b6_g; do train_bird_full "$B"; done
    for B in b4_s1 b5_s2 b6_g; do finish_bird "$B"; done
    echo "[v02] WORKER B ALL DONE $(date +%T)"
  ) > $LOG/bz02_workerB.log 2>&1 &
  local WB=$!
  wait $WA $WB
  echo "[v02] GEN1 COMPLETE $(date +%T)"
}

gen2() {
  # PBT 代际: 生成 gen2 鸟 → 训练+评分 → 代际表
  local CFG2=configs/bizard/v02_gen2.yaml
  until $PY src/models/bizard/v02_pbt.py --select >> $LOG/bz02_pbt_select.log 2>&1; do
    echo "[v02] pbt select retry" >&2; sleep 20
  done
  for B in $($PY -c "import yaml;print(' '.join(b['name'] for b in yaml.safe_load(open('$CFG2'))['birds']))"); do
    CFG=$CFG2 train_bird_full "$B"
    CFG=$CFG2 finish_bird "$B"
  done
  $PY src/models/bizard/v02_pbt.py --report --config $CFG2 >> $LOG/bz02_pbt_report.log 2>&1
  echo "[v02] GEN2 COMPLETE $(date +%T)"
}

case $PHASE in
  gen1) gen1 ;;
  gen2) gen2 ;;
  all)  gen1; gen2 ;;
esac
echo "[v02] PHASE $PHASE DONE $(date +%T)"
