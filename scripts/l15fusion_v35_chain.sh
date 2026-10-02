#!/bin/bash
# fusion v3.5 链: Tahoe 药扰辅助多任务（预案 2b79d02）→ LOCO gate → A/B/C 提交物。
# 幂等: outputs/logs/.v35chain_<name>.done。aux 缓存（tasks/basal）已由冒烟步产出。
# gate: LOCO-HepG2 @λ2.5 ≥ 0.3114（v3 基线 0.3214 − 0.01）；回退则权重 0.15 重试一次（人工触发）。
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
export L15_CFG=configs/l15_fusion/v35.yaml
V3=data/interim/l15_fusion/v3
V35=data/interim/l15_fusion/v35
FLAGDIR=outputs/logs

step() {
  local name=$1; shift
  local flag="$FLAGDIR/.v35chain_${name}.done"
  if [ -f "$flag" ]; then echo "[v35chain] $name already done, skip"; return 0; fi
  local attempt
  for attempt in 1 2; do
    echo "[v35chain] $name start (attempt $attempt) $(date +%T)"
    if "$@"; then touch "$flag"; echo "[v35chain] $name OK $(date +%T)"; return 0; fi
    echo "[v35chain] $name FAILED (attempt $attempt)" >&2; sleep 30
  done
  echo "[v35chain] $name FAILED twice, continue" >&2; return 1
}

# ---- 0) 复用 v3 缓存（vocab/sigstore/tasks/basal/emb；feat_* 与 tpos 必须重建: 22 列布局+aux 行）
step reuse mkdir -p "$V35" && for f in vocab.json \
    sigstore_gwps.npz sigstore_k562_ess.npz sigstore_rpe1.npz sigstore_jurkat.npz sigstore_hepg2.npz sigstore_hct116.npz \
    tasks_gwps.parquet tasks_k562_ess.parquet tasks_rpe1.parquet tasks_jurkat.parquet tasks_hepg2.parquet tasks_hct116.parquet \
    basal_gwps.npy basal_k562_ess.npy basal_rpe1.npy basal_jurkat.npy basal_hepg2.npy \
    basal_a.npy basal_b.npy basal_c.npy \
    emb_se.npy emb_se_meta.json emb_tahoe.npy emb_tahoe_meta.json; do
  [ -f "$V35/$f" ] || cp "$V3/$f" "$V35/$f"
done && rm -f "$V35/feat_train.npz" "$V35/feat_bench_hepg2.npz" "$V35/tpos_train.npy" \
  "$V35/xgb_s35.json" "$V35/mlp_s35.pt" "$V35/xgb_s2_35.json" "$V35/mlp_s2_35_noemb.pt" \
  && ls "$V35" | wc -l

# ---- 1) aux 任务表 + basal reindex（幂等: aux_build_meta.json 存在则跳过由冒烟步覆盖）
step aux $PY src/models/l15_fusion/build_tasks.py --aux-tahoe > logs/v35_aux_tasks.log 2>&1

# ---- 2) 特征装配（主 6 上下文 + aux 行 + feat_bench_hepg2 22 列）
step assemble $PY src/models/l15_fusion/assemble_data.py --which train,bench_hepg2 \
  > logs/v35_assemble.log 2>&1

# ---- 3) stage1 训练（xgb CPU ∥ mlp GPU warm-up 5k 步→混训）
step s1 bash -c '
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
export L15_CFG=configs/l15_fusion/v35.yaml
$PY src/models/l15_fusion/train_xgb.py --stage 1 --model-tag s35 > logs/v35_xgb_s1.log 2>&1 &
P1=$!
$PY src/models/l15_fusion/train_mlp.py --stage 1 --model-tag s35 > logs/v35_mlp_s1.log 2>&1 &
P2=$!
wait $P1 $P2
'

# ---- 4) stage1 LOCO 预测（fused = 0.5·xgb_s35 + 0.5·mlp_s35(emb)）
step s1pred $PY src/models/l15_fusion/predict_loco.py --stage 1 --cache bench_hepg2 \
  --models xgb,mlp --alpha 0.5 --xgb-model xgb_s35 --mlp-pt s35 \
  --tag s1bench_hepg2_v35 --clip 4.0 > logs/v35_predict_loco.log 2>&1

# ---- 5) 并行: LOCO bench λ2.5（gate 臂）∥ stage2 训练
step loco_s2 bash -c '
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
export L15_CFG=configs/l15_fusion/v35.yaml
$PY src/models/l15_fusion/run_bench.py --ctx hepg2 \
  --pred-tag s1bench_hepg2_v35_fused --lam 2.5 --bench-tag l15v35s1_fused_lam2.5 --score \
  > logs/v35s1_bench_lam2.5.log 2>&1 &
P1=$!
$PY src/models/l15_fusion/train_xgb.py --stage 2 --model-tag s2_35 > logs/v35_xgb_s2.log 2>&1 &
P2=$!
$PY src/models/l15_fusion/train_mlp.py --stage 2 --no-emb --model-tag s2_35_noemb > logs/v35_mlp_s2.log 2>&1 &
P3=$!
wait $P1 $P2 $P3
'

# ---- 6) A/B/C 提交物（v3.5 stage2 全池, λ=2.5）
step submit $PY src/models/l15_fusion/build_submission_v1.py \
  --steps predict,stats,build --out-stem 20261003_l15fusion_v35 --lam 2.5 \
  --xgb-tag s2_35 --mlp-pt s2_35_noemb --model-label "l15_fusion v3.5 (tahoe aux multi-task)" \
  > logs/v35_build_submission.log 2>&1

# ---- 7) vcc prep（TMPDIR 大盘）
step vcc bash -c '
set -uo pipefail
mkdir -p /data/resource/wanglab/aivc/tmpdir
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc prep \
  outputs/submissions/20261003_l15fusion_v35.h5ad \
  -g data/raw/controls/gene_names.csv --perts data/raw/controls/pert_counts.csv \
  -o outputs/submissions/20261003_l15fusion_v35.vcc
' > logs/v35_vcc_prep.log 2>&1

echo "[v35chain] ALL DONE $(date +%T)"
