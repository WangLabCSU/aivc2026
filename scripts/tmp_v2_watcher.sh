#!/bin/bash
# v2 接力: 等 stage1 fused Δ̂ → 并行 [LOCO bench λ2.5, λ3.0, stage2 xgb, stage2 mlp]
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
export L15_CFG=configs/l15_fusion/v2.yaml
TARGET=data/interim/l15_fusion/v2/pred_s1bench_hepg2_v2_fused.parquet
while [ ! -f "$TARGET" ]; do sleep 20; done
sleep 5
echo "[watcher] fused parquet ready, firing 4 jobs"
L15_CFG=configs/l15_fusion/v2.yaml $PY src/models/l15_fusion/run_bench.py --ctx hepg2 \
  --pred-tag s1bench_hepg2_v2_fused --lam 2.5 --bench-tag l15v2s1_fused_lam2.5 --score \
  > logs/l15v2s1_bench_lam2.5.log 2>&1 &
P1=$!
L15_CFG=configs/l15_fusion/v2.yaml $PY src/models/l15_fusion/run_bench.py --ctx hepg2 \
  --pred-tag s1bench_hepg2_v2_fused --lam 3.0 --bench-tag l15v2s1_fused_lam3.0 --score \
  > logs/l15v2s1_bench_lam3.0.log 2>&1 &
P2=$!
L15_CFG=configs/l15_fusion/v2.yaml $PY src/models/l15_fusion/train_xgb.py --stage 2 \
  > logs/l15v2_xgb_s2.log 2>&1 &
P3=$!
L15_CFG=configs/l15_fusion/v2.yaml $PY src/models/l15_fusion/train_mlp.py --stage 2 --no-emb \
  > logs/l15v2_mlp_s2.log 2>&1 &
P4=$!
wait $P1 $P2 $P3 $P4
echo "[watcher] all 4 done"
touch data/interim/l15_fusion/v2/PHASE2_DONE
