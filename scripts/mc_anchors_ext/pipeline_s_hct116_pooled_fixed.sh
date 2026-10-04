#!/bin/bash
# S_HCT116 pooled 固定靶集 + 固定 pred 纯 ref 重抽链（单 seed）: baseline → prep-real-bundle → run → score。
# 协议 = pipeline_s_hct116_fixed.sh 逐项对齐（5 指标标度/退化回退/NTC control）, 差异仅:
# ref=ref_pfix_s*（targets.json + min(400,avail)/靶 + NTC2000, 全量 pooled 源）,
# pred=固定共享 l11_lam1.5/S_HCT116_pred.h5ad（轴/扰动集已校验一致, 不随 seed 重建）。
# 用法: bash scripts/mc_anchors_ext/pipeline_s_hct116_pooled_fixed.sh <SEED> <CPUSET>
set -uo pipefail
ROOT=/data/resource/wanglab/aivc/aivc2026
BIN=/data/home/wsx/miniconda3/envs/aivc2026/bin
LB=$ROOT/data/interim/local_bench/S_HCT116
BUNDLE_ROOT=$ROOT/outputs/metrics/local_bench/mc_anchor_bundles
CFG=(--preset vcc2026 --pert-col target_gene --control Non-Targeting
     --input-type counts --set de.backend=pdex)
# 深度 ref 上 expr_mse_unbiased_capped_norm 标度退化 → 沿用 5 指标口径（与 deep/fixed 节点同口径）。
MSET='[pds_cosine, de_wilcoxon_lfc_nmae, de_wilcoxon_direction_fidelity_yield_raw, de_wilcoxon_direction_reach_raw, de_wilcoxon_sig_jaccard]'
CFG5=("${CFG[@]}" --set "metrics=${MSET}")
SEED=$1
CPUS=${2:-0-15}
REF=$LB/ref_pfix_s${SEED}.h5ad
PRED=$ROOT/outputs/metrics/local_bench/l11_lam1.5/S_HCT116_pred.h5ad
BUILD=$LB/build_pfix_s${SEED}
EVALDIR=$LB/eval_pfix_s${SEED}
BUNDLE=$BUNDLE_ROOT/S_HCT116_pooled_fixed_seed${SEED}
SCORED=$LB/scored_pfix_s${SEED}.csv
mkdir -p "$BUILD" "$EVALDIR"
ts() { date "+%H:%M:%S"; }

[[ -f $REF && -f $PRED ]] || { echo "$(ts) ref/pred missing for seed$SEED"; exit 1; }

if [[ ! -f $SCORED ]]; then
  if [[ ! -f $BUNDLE/manifest.json ]]; then
    echo "$(ts) S_HCT116-pooled-fixed seed$SEED step1 baseline (cpu $CPUS)"
    if ! taskset -c "$CPUS" "$BIN/cell-eval2" baseline -ar "$REF" "${CFG5[@]}" \
        -o "$BUILD/baseline" --save-pred "$BUILD/baseline_pred.h5ad" > "$BUILD/baseline.log" 2>&1; then
      echo "$(ts) baseline failed → retry --allow-degenerate-baseline"
      taskset -c "$CPUS" "$BIN/cell-eval2" baseline -ar "$REF" "${CFG5[@]}" \
        -o "$BUILD/baseline" --save-pred "$BUILD/baseline_pred.h5ad" \
        --allow-degenerate-baseline >> "$BUILD/baseline.log" 2>&1 \
        || { echo "$(ts) baseline FAILED seed$SEED"; exit 1; }
    fi
    echo "$(ts) seed$SEED step2 prep-real-bundle (cpu $CPUS)"
    if ! taskset -c "$CPUS" "$BIN/cell-eval2" prep-real-bundle --real "$REF" \
        --baseline "$BUILD/baseline_pred.h5ad" -o "$BUNDLE" --id "S_HCT116_pooled_fixed_seed${SEED}_local" \
        "${CFG5[@]}" > "$BUILD/prep.log" 2>&1; then
      echo "$(ts) prep failed → rebuild baseline with --allow-degenerate-baseline"
      taskset -c "$CPUS" "$BIN/cell-eval2" baseline -ar "$REF" "${CFG5[@]}" \
        -o "$BUILD/baseline" --save-pred "$BUILD/baseline_pred.h5ad" \
        --allow-degenerate-baseline >> "$BUILD/baseline.log" 2>&1
      taskset -c "$CPUS" "$BIN/cell-eval2" prep-real-bundle --real "$REF" \
        --baseline "$BUILD/baseline_pred.h5ad" -o "$BUNDLE" --id "S_HCT116_pooled_fixed_seed${SEED}_local" \
        "${CFG5[@]}" > "$BUILD/prep.log" 2>&1 || { echo "$(ts) prep FAILED seed$SEED"; exit 1; }
    fi
  fi
  echo "$(ts) seed$SEED step3 run fixed pred (cpu $CPUS)"
  taskset -c "$CPUS" "$BIN/cell-eval2" run -ap "$PRED" -ar "$REF" "${CFG5[@]}" \
    -o "$EVALDIR" > "$BUILD/run.log" 2>&1 || { echo "$(ts) run FAILED seed$SEED"; exit 1; }
  echo "$(ts) seed$SEED step4 score vs own bundle"
  taskset -c "$CPUS" "$BIN/cell-eval2" score \
    --user-agg "$EVALDIR/agg_results.csv" --real-bundle "$BUNDLE" \
    -o "$SCORED" > "$BUILD/score.log" 2>&1 || { echo "$(ts) score FAILED seed$SEED"; exit 1; }
fi
echo "$(ts) done S_HCT116-pooled-fixed seed$SEED"
