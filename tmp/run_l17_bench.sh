#!/bin/bash
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
cd /data/resource/wanglab/aivc/aivc2026
$PY src/eval/local_bench.py --model-config configs/l1_signature/l17-lam10-conf.yaml --tag l17_lam1.0_conf --contexts Q_HepG2 > tmp/bench_l17_lam10_conf.log 2>&1 &
$PY src/eval/local_bench.py --model-config configs/l1_signature/l17-lam15-conf-loco.yaml --tag l17_lam1.5_conf_loco --contexts Q_HepG2 > tmp/bench_l17_lam15_conf_loco.log 2>&1 &
wait
echo "bench arms done $(date)"
$PY src/eval/bench_score.py --tags l11_lam1.0,l11_lam1.5,l17_lam1.0_conf,l17_lam1.5_conf_loco --contexts Q_HepG2 --out outputs/metrics/local_bench/confidence_anchor_normalized.json > tmp/score_l17.log 2>&1
echo "SCORING_DONE $(date)"
