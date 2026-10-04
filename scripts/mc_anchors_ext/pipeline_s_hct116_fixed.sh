#!/bin/bash
# S_HCT116 固定靶集深度多重复锚点链（单 seed）: baseline → prep-real-bundle → run pred → score。
# 协议 = pipeline_s_hct116_deep.sh 逐项对齐（5 指标标度/退化回退/NTC control）, 唯一差异:
# 靶集固定（targets_s0.json）+ ref/pred 为 *_fix_* 工件。4 seed 并行时各占 16 核。
# 用法: bash scripts/mc_anchors_ext/pipeline_s_hct116_fixed.sh <SEED> <CPU_A> <CPU_B>
set -uo pipefail
ROOT=/data/resource/wanglab/aivc/aivc2026
BIN=/data/home/wsx/miniconda3/envs/aivc2026/bin
LB=$ROOT/data/interim/local_bench/S_HCT116
BUNDLE_ROOT=$ROOT/outputs/metrics/local_bench/mc_anchor_bundles
CFG=(--preset vcc2026 --pert-col target_gene --control Non-Targeting
     --input-type counts --set de.backend=pdex)
# 深度 ref 上 expr_mse_unbiased_capped_norm 标度退化（replicate 锚点>baseline, 官方拒绝建标度）
# → 沿用深度版 5 指标口径（drop mse）, 与 S_HCT116_deep 节点同口径可比。
MSET='[pds_cosine, de_wilcoxon_lfc_nmae, de_wilcoxon_direction_fidelity_yield_raw, de_wilcoxon_direction_reach_raw, de_wilcoxon_sig_jaccard]'
CFG5=("${CFG[@]}" --set "metrics=${MSET}")
SEED=$1
CPU_A=${2:-0-15}
CPU_B=${3:-16-31}
REF=$LB/ref_fix_s${SEED}.h5ad
PRED=$ROOT/outputs/metrics/local_bench/l11_lam1.5/S_HCT116_pred_fix_s${SEED}.h5ad
BUILD=$LB/build_fix_s${SEED}
EVALDIR=$LB/eval_fix_s${SEED}
BUNDLE=$BUNDLE_ROOT/S_HCT116_fixed_seed${SEED}
SCORED=$LB/scored_fix_s${SEED}.csv
mkdir -p "$BUILD" "$EVALDIR"
ts() { date "+%H:%M:%S"; }

[[ -f $REF && -f $PRED ]] || { echo "$(ts) ref/pred missing for seed$SEED"; exit 1; }

if [[ ! -f $SCORED ]]; then
  if [[ ! -f $BUNDLE/manifest.json ]]; then
    echo "$(ts) S_HCT116-fixed seed$SEED step1 baseline (cpu $CPU_A)"
    if ! taskset -c "$CPU_A" "$BIN/cell-eval2" baseline -ar "$REF" "${CFG5[@]}" \
        -o "$BUILD/baseline" --save-pred "$BUILD/baseline_pred.h5ad" > "$BUILD/baseline.log" 2>&1; then
      echo "$(ts) baseline failed → retry --allow-degenerate-baseline"
      taskset -c "$CPU_A" "$BIN/cell-eval2" baseline -ar "$REF" "${CFG5[@]}" \
        -o "$BUILD/baseline" --save-pred "$BUILD/baseline_pred.h5ad" \
        --allow-degenerate-baseline >> "$BUILD/baseline.log" 2>&1 \
        || { echo "$(ts) baseline FAILED seed$SEED"; exit 1; }
    fi
    echo "$(ts) seed$SEED step2 prep-real-bundle (cpu $CPU_A)"
    if ! taskset -c "$CPU_A" "$BIN/cell-eval2" prep-real-bundle --real "$REF" \
        --baseline "$BUILD/baseline_pred.h5ad" -o "$BUNDLE" --id "S_HCT116_fixed_seed${SEED}_local" \
        "${CFG5[@]}" > "$BUILD/prep.log" 2>&1; then
      echo "$(ts) prep failed → rebuild baseline with --allow-degenerate-baseline"
      taskset -c "$CPU_A" "$BIN/cell-eval2" baseline -ar "$REF" "${CFG5[@]}" \
        -o "$BUILD/baseline" --save-pred "$BUILD/baseline_pred.h5ad" \
        --allow-degenerate-baseline >> "$BUILD/baseline.log" 2>&1
      taskset -c "$CPU_A" "$BIN/cell-eval2" prep-real-bundle --real "$REF" \
        --baseline "$BUILD/baseline_pred.h5ad" -o "$BUNDLE" --id "S_HCT116_fixed_seed${SEED}_local" \
        "${CFG5[@]}" > "$BUILD/prep.log" 2>&1 || { echo "$(ts) prep FAILED seed$SEED"; exit 1; }
    fi
  fi
  echo "$(ts) seed$SEED step3 run candidate pred (cpu $CPU_B)"
  taskset -c "$CPU_B" "$BIN/cell-eval2" run -ap "$PRED" -ar "$REF" "${CFG5[@]}" \
    -o "$EVALDIR" > "$BUILD/run.log" 2>&1 || { echo "$(ts) run FAILED seed$SEED"; exit 1; }
  echo "$(ts) seed$SEED step4 score vs own bundle"
  taskset -c "$CPU_B" "$BIN/cell-eval2" score \
    --user-agg "$EVALDIR/agg_results.csv" --real-bundle "$BUNDLE" \
    -o "$SCORED" > "$BUILD/score.log" 2>&1 || { echo "$(ts) score FAILED seed$SEED"; exit 1; }
fi
echo "$(ts) done S_HCT116-fixed seed$SEED"
