#!/bin/bash
# E3 单 (context, seed) 锚点链: baseline → prep-real-bundle → run → score（协议 = /tmp/mc_anchors/pipeline.sh）
set -e
CTX=$1; SEED=$2; CORES=$3
BIN=/data/home/wsx/miniconda3/envs/aivc2026/bin
ROOT=/data/resource/wanglab/aivc/aivc2026
# /tmp 仅 14G 禁用 → 默认重定向到 /data（env MC_ANCHORS_TMP 可覆盖）
TMP=${MC_ANCHORS_TMP:-$ROOT/data/interim/mc_anchors_ext}
REF=$TMP/${CTX}_seed${SEED}.h5ad
BUILD=$TMP/build_${CTX}_seed${SEED}
EVALDIR=$TMP/eval_${CTX}_seed${SEED}
BUNDLE=$ROOT/outputs/metrics/local_bench/mc_anchor_bundles/${CTX}_seed${SEED}
PRED=$ROOT/outputs/metrics/local_bench/l11_lam1.5/${CTX}_pred.h5ad
CFG=(--preset vcc2026 --pert-col target_gene --control non-targeting
     --input-type counts --set de.backend=pdex)
mkdir -p "$BUILD" "$EVALDIR"
ts() { date "+%H:%M:%S"; }
echo "[$ts] $CTX seed$SEED step1 baseline"
taskset -c "$CORES" "$BIN/cell-eval2" baseline -ar "$REF" "${CFG[@]}" \
  -o "$BUILD/baseline" --save-pred "$BUILD/baseline_pred.h5ad" > "$BUILD/baseline.log" 2>&1
echo "[$ts] step2 prep-real-bundle"
taskset -c "$CORES" "$BIN/cell-eval2" prep-real-bundle --real "$REF" \
  --baseline "$BUILD/baseline_pred.h5ad" -o "$BUNDLE" --id "${CTX}_seed${SEED}_local" \
  "${CFG[@]}" > "$BUILD/prep.log" 2>&1
echo "[$ts] step3 run candidate pred"
taskset -c "$CORES" "$BIN/cell-eval2" run -ap "$PRED" -ar "$REF" "${CFG[@]}" \
  -o "$EVALDIR" > "$BUILD/run.log" 2>&1
echo "[$ts] step4 score vs own bundle"
taskset -c "$CORES" "$BIN/cell-eval2" score \
  --user-agg "$EVALDIR/agg_results.csv" --real-bundle "$BUNDLE" \
  -o "$TMP/${CTX}_seed${SEED}_scored.csv" > "$BUILD/score.log" 2>&1
echo "[$ts] done $CTX seed$SEED"
