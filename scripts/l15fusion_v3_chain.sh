#!/bin/bash
# fusion v3 夜间链: 深度 hct116 签名 → 训练池升级重训 → LOCO + A/B/C 提交物 + S_HCT116 fid 专项。
# 幂等: 每步成功写 outputs/logs/.v3chain_<name>.done；重跑跳过已完成步。
# 前置: data/interim/signatures/hct116_all.parquet 为 109 批深版 (build_orion_signatures_v3.py)。
# vocab 冻结 = v2 vocab（拷贝缓存），v3-vs-v2 LOCO 差异可归因于纯签名深度。
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
export L15_CFG=configs/l15_fusion/v3.yaml
V2=data/interim/l15_fusion/v2
V3=data/interim/l15_fusion/v3
FLAGDIR=outputs/logs

step() {
  local name=$1; shift
  local flag="$FLAGDIR/.v3chain_${name}.done"
  if [ -f "$flag" ]; then echo "[v3chain] $name already done, skip"; return 0; fi
  local attempt
  for attempt in 1 2; do
    echo "[v3chain] $name start (attempt $attempt) $(date +%T)"
    if "$@"; then touch "$flag"; echo "[v3chain] $name OK $(date +%T)"; return 0; fi
    echo "[v3chain] $name FAILED (attempt $attempt)" >&2; sleep 30
  done
  echo "[v3chain] $name FAILED twice, continue" >&2; return 1
}

# ---- 0) workdir 缓存复用（vocab/basal/emb/其他上下文 sigstore+tasks 全部与 v2 逐字节一致）
step reuse mkdir -p "$V3" && for f in vocab.json \
    sigstore_gwps.npz sigstore_k562_ess.npz sigstore_rpe1.npz sigstore_jurkat.npz sigstore_hepg2.npz \
    tasks_gwps.parquet tasks_k562_ess.parquet tasks_rpe1.parquet tasks_jurkat.parquet tasks_hepg2.parquet \
    basal_gwps.npy basal_k562_ess.npy basal_rpe1.npy basal_jurkat.npy basal_hepg2.npy \
    basal_a.npy basal_b.npy basal_c.npy \
    emb_se.npy emb_se_meta.json emb_tahoe.npy emb_tahoe_meta.json; do
  [ -f "$V3/$f" ] || cp "$V2/$f" "$V3/$f"
done && ls "$V3" | wc -l

# ---- 1) 特征装配（hct116 深度 sigstore+tasks 重建 + feat_train + feat_bench_hepg2）
step assemble $PY src/models/l15_fusion/assemble_data.py --which train,bench_hepg2 \
  > logs/v3_assemble.log 2>&1

# ---- 2) stage1 训练（xgb CPU ∥ mlp GPU）
step s1 bash -c '
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
export L15_CFG=configs/l15_fusion/v3.yaml
$PY src/models/l15_fusion/train_xgb.py --stage 1 > logs/v3_xgb_s1.log 2>&1 &
P1=$!
$PY src/models/l15_fusion/train_mlp.py --stage 1 > logs/v3_mlp_s1.log 2>&1 &
P2=$!
wait $P1 $P2
'

# ---- 3) stage1 LOCO 预测（fused = 0.5·xgb_s1 + 0.5·mlp_s1(emb)，与 v2 完全同构）
step s1pred $PY src/models/l15_fusion/predict_loco.py --stage 1 --cache bench_hepg2 \
  --models xgb,mlp --alpha 0.5 --tag s1bench_hepg2_v3 --clip 4.0 \
  > logs/v3_predict_loco.log 2>&1

# ---- 4) 并行: LOCO bench λ2.5/λ3.0 (CPU cell-eval2) ∥ stage2 训练 (xgb CPU ∥ mlp GPU)
step loco_s2 bash -c '
set -uo pipefail
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
export L15_CFG=configs/l15_fusion/v3.yaml
$PY src/models/l15_fusion/run_bench.py --ctx hepg2 \
  --pred-tag s1bench_hepg2_v3_fused --lam 2.5 --bench-tag l15v3s1_fused_lam2.5 --score \
  > logs/v3s1_bench_lam2.5.log 2>&1 &
P1=$!
$PY src/models/l15_fusion/run_bench.py --ctx hepg2 \
  --pred-tag s1bench_hepg2_v3_fused --lam 3.0 --bench-tag l15v3s1_fused_lam3.0 --score \
  > logs/v3s1_bench_lam3.0.log 2>&1 &
P2=$!
$PY src/models/l15_fusion/train_xgb.py --stage 2 > logs/v3_xgb_s2.log 2>&1 &
P3=$!
$PY src/models/l15_fusion/train_mlp.py --stage 2 --no-emb > logs/v3_mlp_s2.log 2>&1 &
P4=$!
wait $P1 $P2 $P3 $P4
'

# ---- 5) A/B/C 提交物（stage2 全池, λ=2.5 官方已验证锚点）
step submit $PY src/models/l15_fusion/build_submission_v1.py \
  --steps predict,stats,build --out-stem 20261003_l15fusion_v3 --lam 2.5 \
  > logs/v3_build_submission.log 2>&1

# ---- 6) vcc prep 全绿
step vcc /data/home/wsx/miniconda3/envs/aivc2026/bin/vcc prep \
  outputs/submissions/20261003_l15fusion_v3.h5ad \
  --perts data/raw/controls/pert_counts.csv \
  -o outputs/submissions/20261003_l15fusion_v3.vcc \
  > logs/v3_vcc_prep.log 2>&1

# ---- 7) S_HCT116 fid 专项: v3 stage2 @λ2.5（Δ̂ 缓存）与 @λ1.5（同 Δ̂ 重建载体）
step s_hct116 $PY src/models/l15_fusion/predict_hct116_bench.py \
  --cfg configs/l15_fusion/v3.yaml --lam 2.5 \
  --pred-tag s2bench_hct116_v3 --bench-tag l15v3_hct116_lam2.5 \
  > logs/v3_hct116_lam2.5.log 2>&1
step s_hct116_lam15 $PY src/models/l15_fusion/predict_hct116_bench.py \
  --cfg configs/l15_fusion/v3.yaml --lam 1.5 \
  --pred-tag s2bench_hct116_v3 --bench-tag l15v3_hct116_lam1.5 \
  > logs/v3_hct116_lam1.5.log 2>&1

# ---- 8) S_HCT116 归一化评分（bundle 已就绪）
step s_score $PY src/eval/bench_score.py \
  --tags l15v3_hct116_lam2.5,l15v3_hct116_lam1.5,l15v2_hct116_lam1.5,l11_lam1.5 \
  --contexts S_HCT116 \
  --out outputs/metrics/local_bench/hct116_v3_norm.json \
  > logs/v3_s_score.log 2>&1

# ---- 9) 启发式 λ3.5 对照: Q_HepG2（LOCO 对照组补测）+ S_HCT116（fid 专项对照）
step heur_q $PY src/eval/local_bench.py \
  --model-config configs/l1_signature/l17-lam35-dailybest.yaml --tag l17_lam35 --contexts Q_HepG2 \
  > logs/v3_heur_qhepg2.log 2>&1
step heur_q_score $PY src/eval/bench_score.py --tags l17_lam35 --contexts Q_HepG2 \
  --out outputs/metrics/local_bench/l17_lam35/anchor_normalized.json \
  > logs/v3_heur_q_score.log 2>&1
step heur_s $PY src/eval/local_bench.py \
  --model-config configs/l1_signature/l17-lam35-dailybest.yaml --tag l17_lam35 --contexts S_HCT116 \
  > logs/v3_heur_shct116.log 2>&1
step heur_s_score $PY src/eval/bench_score.py --tags l17_lam35 --contexts S_HCT116 \
  --out outputs/metrics/local_bench/l17_lam35/anchor_normalized_S.json \
  > logs/v3_heur_s_score.log 2>&1

echo "[v3chain] ALL DONE $(date +%T)"
