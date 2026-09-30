#!/bin/bash
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
cd /data/resource/wanglab/aivc/aivc2026
$PY src/eval/local_bench.py --model-config configs/l1_signature/l17-lam15-conf-inv.yaml --tag l17_lam1.5_conf_inv --contexts Q_HepG2 > tmp/bench_l17_inv_q.log 2>&1 &
$PY src/eval/local_bench.py --model-config configs/l1_signature/l17-lam15-conf-p1.yaml --tag l17_lam1.5_conf_p1 --contexts P_H1 > tmp/bench_l17_p1.log 2>&1 &
$PY src/eval/local_bench.py --model-config configs/l1_signature/l17-lam15-conf-inv-p1.yaml --tag l17_lam1.5_conf_inv_p1 --contexts P_H1 > tmp/bench_l17_inv_p1.log 2>&1 &
wait
echo "diag arms done $(date)"
$PY src/eval/bench_score.py --tags l17_lam1.5_conf_inv,l17_lam1.5_conf_p1,l17_lam1.5_conf_inv_p1 --contexts Q_HepG2,P_H1 --out outputs/metrics/local_bench/confidence_diag_normalized.json > tmp/score_l17_diag.log 2>&1
echo "DIAG_SCORING_DONE $(date)"
