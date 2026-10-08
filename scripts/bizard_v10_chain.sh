#!/bin/bash
# BizardNet v1.0 评估+提交链: LOCO 评估(bench Q_HepG2 + proxy) → full 臂 → A/B/C 提交物
set -e
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
WN=data/interim/bizard/bizardnet

echo "=== [1] LOCO proxy eval on hepg2 rows ==="
$PY src/models/bizard/bizardnet.py eval-loco --arm loco

echo "=== [2] LOCO bench predict (bench_hepg2) ==="
$PY src/models/bizard/bizardnet.py predict --src bench_hepg2 --arm loco

echo "=== [3] Q_HepG2 bench @lam2.5 (anchor normalized) ==="
$PY src/models/bizard/bench_bird.py --tag bzbiz10_loco_lam2.5 --lam 2.5 --contexts Q_HepG2 \
  --delta-dir $WN --delta-pattern "pred_bizardnet_loco_bench_hepg2.parquet"

echo "=== [4] full arm train ==="
$PY src/models/bizard/bizardnet.py train --arm full

echo "=== [5] submit predict A/B/C ==="
for cx in A B C; do
  $PY src/models/bizard/bizardnet.py predict --src submit_$cx --arm full
done

echo "=== [6] per-context centering → delta_submit ==="
$PY src/models/bizard/bizardnet_submit_deltas.py

echo "=== [7] build submission h5ad (lam3.5) ==="
$PY src/models/bizard/build_submission.py --lam 3.5 \
  --config configs/bizard/bizardnet.yaml \
  --delta-pattern "delta_submit_{cx}.parquet" \
  --out-stem 20261008_bizard_v10 \
  --model-label "bizard v1.0 BizardNet transformer (ctx-cond, seq-genes)"

echo "=== [8] vcc prep ==="
TMPDIR=/data/resource/wanglab/aivc/tmpdir /data/home/wsx/miniconda3/envs/aivc2026/bin/vcc prep \
  outputs/submissions/20261008_bizard_v10.h5ad \
  -g data/raw/controls/gene_names.csv --perts data/raw/controls/pert_counts.csv \
  -o outputs/submissions/20261008_bizard_v10.vcc

echo "=== chain complete ==="
