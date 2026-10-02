#!/bin/bash
# HCT116 全深度 fid 坍塌检验链 v2（幂等 + 单步重试一次 + 聚合/汇总/commit）
# 用法: nohup bash scripts/hct116_fulldepth_chain.sh > outputs/logs/hct116_chain.log 2>&1 &
# 幂等: 每步成功后写 outputs/logs/.chain_<name>.done；重跑跳过已完成步。
# 下载 marker: data/public/xatlas/.orion_all.done
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
FLAGDIR=outputs/logs
DONE_ALL=data/public/xatlas/.hct116_chain_done

step() { # step <name> <cmd...>
  local name=$1; shift
  local flag="$FLAGDIR/.chain_${name}.done"
  if [ -f "$flag" ]; then echo "[chain] $name already done, skip"; return 0; fi
  local attempt
  for attempt in 1 2; do
    echo "[chain] $name start (attempt $attempt) $(date +%T)"
    if "$@"; then touch "$flag"; echo "[chain] $name OK $(date +%T)"; return 0; fi
    echo "[chain] $name FAILED (attempt $attempt), retrying after 60s" >&2
    sleep 60
  done
  echo "[chain] $name FAILED twice, continue to next step" >&2
  return 1
}

echo "[chain] waiting for .orion_all.done ..."
while [ ! -f data/public/xatlas/.orion_all.done ]; do sleep 120; done
echo "[chain] download complete $(date +%T)"

# 1) pooled 全批次重建（幂等: 批次数/细胞数不变则跳过；原子写；向量化）
step pooled $PY src/data/build_pooled_hct116.py

# 2) ref 重建（400/2000 细胞帽, min30 gate）+ 启发式 λ1.5 对照（同 ref）+ 清陈旧产物
step ref_heuristic bash -c '
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
rm -rf data/interim/local_bench/S_HCT116
rm -rf outputs/metrics/local_bench/bundles/S_HCT116 outputs/metrics/local_bench/bundle_build/S_HCT116*
rm -f outputs/metrics/local_bench/l11_lam1.5/S_HCT116_* \
      outputs/metrics/local_bench/l15v1_hct116_lam2.5/S_HCT116_* \
      outputs/metrics/local_bench/l15v2_hct116_lam1.5/S_HCT116_* \
      outputs/metrics/local_bench/l15v2_hct116_lam2.5/S_HCT116_*
# 新 ref 的 100 靶重采样 → 旧靶 Δ̂ 缓存必须清除（否则新靶全部 Δ=0）
rm -f data/interim/l15_fusion/v2/pred_s2bench_hct116_v2.parquet \
      data/interim/l15_fusion/v1/pred_s2bench_hct116_v1.parquet
/data/home/wsx/miniconda3/envs/aivc2026/bin/python src/eval/local_bench.py \
  --model-config configs/l1_signature/ws-v1.2-sampled-lam15.yaml \
  --tag l11_lam1.5 --contexts S_HCT116
'

# 3) fusion v2 (λ2.5 + λ1.5) 与 fusion v1(LOCO 消融) 预测+评测（并行, 各自重试一次）
runjob() { # runjob <name> <cmd...>
  local name=$1; shift
  local attempt
  for attempt in 1 2; do
    if "$@"; then echo "[chain] job $name OK (attempt $attempt)"; return 0; fi
    echo "[chain] job $name FAILED (attempt $attempt)" >&2; sleep 60
  done
  echo "[chain] job $name FAILED twice" >&2; return 1
}
( runjob fusion_v2_lam25 $PY src/models/l15_fusion/predict_hct116_bench.py \
    --cfg configs/l15_fusion/v2.yaml --lam 2.5 \
    --pred-tag s2bench_hct116_v2 --bench-tag l15v2_hct116_lam2.5 \
    > outputs/logs/chain_fusion_v2_lam25.log 2>&1 ) &
P1=$!
( runjob fusion_v2_lam15 $PY src/models/l15_fusion/predict_hct116_bench.py \
    --cfg configs/l15_fusion/v2.yaml --lam 1.5 \
    --pred-tag s2bench_hct116_v2 --bench-tag l15v2_hct116_lam1.5 \
    > outputs/logs/chain_fusion_v2_lam15.log 2>&1 ) &
P2=$!
( runjob fusion_v1_lam25 $PY src/models/l15_fusion/predict_hct116_bench.py \
    --cfg configs/l15_fusion/v1h.yaml --lam 2.5 \
    --pred-tag s2bench_hct116_v1 --bench-tag l15v1_hct116_lam2.5 \
    > outputs/logs/chain_fusion_v1_lam25.log 2>&1 ) &
P3=$!
wait $P1 $P2 $P3
touch $FLAGDIR/.chain_fusion_evals.done
echo "[chain] fusion evals done $(date +%T)"

# 4) bundle（--allow-degenerate-baseline; prep 拒收时自动 manual anchor 5 指标口径）+ 归一化评分
step score $PY src/eval/bench_score.py \
  --tags l11_lam1.5,l15v2_hct116_lam2.5,l15v2_hct116_lam1.5,l15v1_hct116_lam2.5 \
  --contexts S_HCT116 --build-bundles --allow-degenerate-baseline \
  --out outputs/metrics/local_bench/hct116_fulldepth_norm.json

# 5) 聚合 → hct116_fulldepth.json + fusion_v2_summary.md 追加小节 + git commit
step aggregate $PY scripts/hct116_fulldepth_aggregate.py --commit

touch $DONE_ALL
echo "[chain] ALL DONE $(date +%T)"
