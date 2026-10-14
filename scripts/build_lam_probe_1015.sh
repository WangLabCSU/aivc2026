#!/bin/bash
# iter021: 10-15 UTC 双发前置 — λ37/λ35 下探构建（bizard-v0.2 家族冻结链）
# 模板: build_lam47_v02.sh; 仅换 --lam 与 STEM; bench 非阻塞; 本脚本只产 artifact 不提交
set -u
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc
export TMPDIR=/data/resource/wanglab/aivc/tmpdir

for LAM in 37 35; do
  STEM=20261015_bizard_v02_lam${LAM}
  echo "=== [lam$LAM] build h5ad (base: bizardnet delta_submit A/B/C) ==="
  $PY src/models/bizard/build_submission.py --lam $LAM \
    --config configs/bizard/bizardnet.yaml \
    --delta-pattern "delta_submit_{cx}.parquet" \
    --out-stem $STEM \
    --model-label "bizard v1.0 BizardNet skip-carrier" || { echo "[FAIL] build lam$LAM"; continue; }
  echo "=== [lam$LAM] vcc prep ==="
  $VCC prep outputs/submissions/$STEM.h5ad \
    -g data/raw/controls/gene_names.csv --perts data/raw/controls/pert_counts.csv \
    -o outputs/submissions/$STEM.vcc || { echo "[FAIL] prep lam$LAM"; continue; }
  echo "=== [lam$LAM] local bench Q_HepG2 @lam$LAM (非阻塞) ==="
  $PY src/models/bizard/bench_bird.py --tag bzbiz10skip_lam${LAM} --lam $LAM --contexts Q_HepG2 \
    --delta-dir data/interim/bizard/bizardnet --delta-pattern "pred_skip_bench_hepg2.parquet" || \
    echo "[warn] bench_bird lam$LAM failed (非阻塞 — 提交决策仍按 R-1 默认)"
done
echo "=== probe chain complete: outputs/submissions/20261015_bizard_v02_lam{37,35}.vcc ==="
