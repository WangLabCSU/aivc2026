#!/bin/bash
# E5 S_HCT116 多重复锚点链（seed 11/12/13）: baseline → prep-real-bundle → run pred → score。
# 协议 = mc_anchors_ext/pipeline.sh，差异: --control Non-Targeting；baseline 退化时
# 回退 --allow-degenerate-baseline 重跑。工件落 $MC_ANCHORS_TMP(默认 data/interim/mc_anchors_ext)。
set -uo pipefail
ROOT=/data/resource/wanglab/aivc/aivc2026
BIN=/data/home/wsx/miniconda3/envs/aivc2026/bin
TMP=${MC_ANCHORS_TMP:-$ROOT/data/interim/mc_anchors_ext}
BUNDLE_ROOT=$ROOT/outputs/metrics/local_bench/mc_anchor_bundles
PRED=$ROOT/outputs/metrics/local_bench/l11_lam1.5/S_HCT116_pred.h5ad
CFG=(--preset vcc2026 --pert-col target_gene --control Non-Targeting
     --input-type counts --set de.backend=pdex)
SEED=$1
REF=$TMP/S_HCT116_seed${SEED}.h5ad
BUILD=$TMP/build_S_HCT116_seed${SEED}
EVALDIR=$TMP/eval_S_HCT116_seed${SEED}
BUNDLE=$BUNDLE_ROOT/S_HCT116_seed${SEED}
mkdir -p "$BUILD" "$EVALDIR"
ts() { date "+%H:%M:%S"; }

echo "$(date "+%H:%M:%S") S_HCT116 seed$SEED step1 baseline"
if ! taskset -c 0-15 "$BIN/cell-eval2" baseline -ar "$REF" "${CFG[@]}" \
    -o "$BUILD/baseline" --save-pred "$BUILD/baseline_pred.h5ad" > "$BUILD/baseline.log" 2>&1; then
  echo "$(date "+%H:%M:%S") baseline failed → retry --allow-degenerate-baseline"
  taskset -c 0-15 "$BIN/cell-eval2" baseline -ar "$REF" "${CFG[@]}" \
    -o "$BUILD/baseline" --save-pred "$BUILD/baseline_pred.h5ad" \
    --allow-degenerate-baseline >> "$BUILD/baseline.log" 2>&1 || { echo "$(date "+%H:%M:%S") baseline FAILED"; exit 1; }
fi
echo "$(date "+%H:%M:%S") step2 prep-real-bundle"
if ! taskset -c 0-15 "$BIN/cell-eval2" prep-real-bundle --real "$REF" \
    --baseline "$BUILD/baseline_pred.h5ad" -o "$BUNDLE" --id "S_HCT116_seed${SEED}_local" \
    "${CFG[@]}" > "$BUILD/prep.log" 2>&1; then
  echo "$(date "+%H:%M:%S") prep failed → rebuild baseline with --allow-degenerate-baseline"
  taskset -c 0-15 "$BIN/cell-eval2" baseline -ar "$REF" "${CFG[@]}" \
    -o "$BUILD/baseline" --save-pred "$BUILD/baseline_pred.h5ad" \
    --allow-degenerate-baseline >> "$BUILD/baseline.log" 2>&1
  taskset -c 0-15 "$BIN/cell-eval2" prep-real-bundle --real "$REF" \
    --baseline "$BUILD/baseline_pred.h5ad" -o "$BUNDLE" --id "S_HCT116_seed${SEED}_local" \
    "${CFG[@]}" > "$BUILD/prep.log" 2>&1 || { echo "$(date "+%H:%M:%S") prep FAILED"; exit 1; }
fi
echo "$(date "+%H:%M:%S") step3 run candidate pred"
taskset -c 16-31 "$BIN/cell-eval2" run -ap "$PRED" -ar "$REF" "${CFG[@]}" \
  -o "$EVALDIR" > "$BUILD/run.log" 2>&1 || { echo "$(date "+%H:%M:%S") run FAILED"; exit 1; }
echo "$(date "+%H:%M:%S") step4 score vs own bundle"
taskset -c 16-31 "$BIN/cell-eval2" score \
  --user-agg "$EVALDIR/agg_results.csv" --real-bundle "$BUNDLE" \
  -o "$TMP/S_HCT116_seed${SEED}_scored.csv" > "$BUILD/score.log" 2>&1 || { echo "$(date "+%H:%M:%S") score FAILED"; exit 1; }
echo "$(date "+%H:%M:%S") done S_HCT116 seed$SEED"
