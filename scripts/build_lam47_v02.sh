#!/bin/bash
# iter021: slot2 (10-14 UTC 剩余1发) — λ47 bizard-v0.2
# 路径: build_submission.py --lam 47 (复用 λ39 delta_submit_{A,B,C}.parquet, 同 λ55 构建法)
# 出 h5ad → vcc prep → 本地 bench Q_HepG2 (λ47 期望校准点) → 人工确认后提交
set -e
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
STEM=20261014_bizard_v02_lam47

echo "=== [1] build h5ad lam=47 (base: bizardnet delta_submit A/B/C) ==="
$PY src/models/bizard/build_submission.py --lam 47 \
  --config configs/bizard/bizardnet.yaml \
  --delta-pattern "delta_submit_{cx}.parquet" \
  --out-stem $STEM \
  --model-label "bizard v1.0 BizardNet skip-carrier"

echo "=== [2] vcc prep ==="
$VCC prep outputs/submissions/$STEM.h5ad \
  -g data/raw/controls/gene_names.csv --perts data/raw/controls/pert_counts.csv \
  -o outputs/submissions/$STEM.vcc

echo "=== [3] local bench Q_HepG2 @lam47 artifact ==="
$PY src/models/bizard/bench_bird.py --tag bzbiz10skip_lam47 --lam 47 --contexts Q_HepG2 \
  --delta-dir data/interim/bizard/bizardnet --delta-pattern "delta_submit_{cx}.parquet" || \
  echo "[warn] bench_bird failed (非阻塞 — 提交决策仍按 R-1 默认)"

echo "=== chain complete: outputs/submissions/$STEM.vcc ==="
