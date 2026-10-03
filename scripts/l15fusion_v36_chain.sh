#!/bin/bash
# fusion v3.6 链: Tahoe 药扰 aux 扩展到全 50 细胞系（4,700 组全用, w=0.15, 每系 DMSO_TF basal）
#   → LOCO-HepG2 gate → A/B/C 提交物。
# 复用 v35 缓存（vocab/sigstore/tasks 主 6 ctx/basal/emb）；feat_* 与 aux 任务表重建。
# gate: LOCO-HepG2 @λ2.5 ≥ 0.3213（v3.5 基线 0.3313 − 0.01）；不过则不打包提交物。
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
export L15_CFG=configs/l15_fusion/v36.yaml
V35=data/interim/l15_fusion/v35
V36=data/interim/l15_fusion/v36
FLAGDIR=outputs/logs
GATE=0.3213
BENCH_TAG=l15v36aux50s1_fused_lam2.5

step() {
  local name=$1; shift
  local flag="$FLAGDIR/.v36chain_${name}.done"
  if [ -f "$flag" ]; then echo "[v36chain] $name already done, skip"; return 0; fi
  local attempt
  for attempt in 1 2; do
    echo "[v36chain] $name start (attempt $attempt) $(date +%T)"
    if "$@"; then touch "$flag"; echo "[v36chain] $name OK $(date +%T)"; return 0; fi
    echo "[v36chain] $name FAILED (attempt $attempt)" >&2; sleep 10
  done
  echo "[v36chain] $name FAILED twice, ABORT" >&2; return 1
}

# ---- 0) 等 basal 提取完成（50 系 raw npy）
step wait_basal bash -c "
while [ ! -f $V36/basal_extract_meta.json ]; do sleep 20; done
ls $V36/raw_basal_*.npy | wc -l"

# ---- 1) 复用 v35 缓存（主任务部分与 v3.5 完全一致）
step reuse bash -c 'for f in vocab.json \
    sigstore_gwps.npz sigstore_k562_ess.npz sigstore_rpe1.npz sigstore_jurkat.npz sigstore_hepg2.npz sigstore_hct116.npz \
    tasks_gwps.parquet tasks_k562_ess.parquet tasks_rpe1.parquet tasks_jurkat.parquet tasks_hepg2.parquet tasks_hct116.parquet \
    basal_gwps.npy basal_k562_ess.npy basal_rpe1.npy basal_jurkat.npy basal_hepg2.npy \
    basal_a.npy basal_b.npy basal_c.npy \
    emb_se.npy emb_se_meta.json emb_tahoe.npy emb_tahoe_meta.json; do
  [ -f "'"$V36"'/$f" ] || cp "'"$V35"'/$f" "'"$V36"'/$f"
done && ls '"$V36"' | wc -l'

# ---- 2) aux 任务表（50 系×94 组, top500+rand500）+ basal reindex（vocab 序 log1p-CPM）
step aux $PY src/models/l15_fusion/build_tasks.py --aux-tahoe > logs/v36_aux_tasks.log 2>&1

# ---- 3) 特征装配（主 6 上下文 30.8M + aux ~4.7M 行 + feat_bench_hepg2）
step assemble $PY src/models/l15_fusion/assemble_data.py --which train,bench_hepg2 \
  > logs/v36_assemble.log 2>&1

# ---- 4) stage1 训练（xgb CPU ∥ mlp GPU）
step s1 bash -c '
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
export L15_CFG=configs/l15_fusion/v36.yaml
$PY src/models/l15_fusion/train_xgb.py --stage 1 --model-tag s36 > logs/v36_xgb_s1.log 2>&1 &
P1=$!
$PY src/models/l15_fusion/train_mlp.py --stage 1 --model-tag s36 > logs/v36_mlp_s1.log 2>&1 &
P2=$!
wait $P1 $P2
'

# ---- 5) stage1 LOCO 预测（fused = 0.5·xgb_s36 + 0.5·mlp_s36(emb)）
step s1pred $PY src/models/l15_fusion/predict_loco.py --stage 1 --cache bench_hepg2 \
  --models xgb,mlp --alpha 0.5 --xgb-model xgb_s36 --mlp-pt s36 \
  --tag s1bench_hepg2_v36 --clip 4.0 > logs/v36_predict_loco.log 2>&1

# ---- 6) 并行: LOCO bench λ2.5（gate 臂）∥ stage2 训练
step loco_s2 bash -c '
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
export L15_CFG=configs/l15_fusion/v36.yaml
$PY src/models/l15_fusion/run_bench.py --ctx hepg2 \
  --pred-tag s1bench_hepg2_v36_fused --lam 2.5 --bench-tag '"$BENCH_TAG"' --score \
  > logs/v36s1_bench_lam2.5.log 2>&1 &
P1=$!
$PY src/models/l15_fusion/train_xgb.py --stage 2 --model-tag s2_36 > logs/v36_xgb_s2.log 2>&1 &
P2=$!
$PY src/models/l15_fusion/train_mlp.py --stage 2 --no-emb --model-tag s2_36_noemb > logs/v36_mlp_s2.log 2>&1 &
P3=$!
wait $P1 $P2 $P3
'

# ---- 7) gate: LOCO-HepG2 overall ≥ 0.3213
step gate $PY -c "
import json, sys
j = json.load(open('outputs/metrics/local_bench/$BENCH_TAG/anchor_normalized.json'))
o = j['per_tag']['$BENCH_TAG']['overall']['overall']
print(f'LOCO-HepG2 overall = {o:.4f} (gate >= $GATE)')
sys.exit(0 if o >= $GATE else 1)"

# ---- 8) A/B/C 提交物（stage2 全池, λ=2.5）
step submit $PY src/models/l15_fusion/build_submission_v1.py \
  --steps predict,stats,build --out-stem 20261003_l15fusion_v36 --lam 2.5 \
  --xgb-tag s2_36 --mlp-pt s2_36_noemb --model-label "l15_fusion v3.6 (tahoe aux 50-cell-line multi-task)" \
  > logs/v36_build_submission.log 2>&1

# ---- 9) vcc prep（TMPDIR 大盘）
step vcc bash -c '
set -uo pipefail
mkdir -p /data/resource/wanglab/aivc/tmpdir
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc prep \
  outputs/submissions/20261003_l15fusion_v36.h5ad \
  -g data/raw/controls/gene_names.csv --perts data/raw/controls/pert_counts.csv \
  -o outputs/submissions/20261003_l15fusion_v36.vcc
' > logs/v36_vcc_prep.log 2>&1

echo "[v36chain] ALL DONE $(date +%T)"
