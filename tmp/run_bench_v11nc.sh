#!/bin/bash
# iter033 v0.11 NC (sign-noncrossing 中心化) 链: bench → score → heur predict
# sim4 证据: NC_b1.0 pds +0.0501 ≥ HCX +0.0454 且 fid 代理 dF=+0.0000
# (不用 set -u: conda deactivate 钩子 unbound variable 会误杀)
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
cd /data/resource/wanglab/aivc/aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir

echo "[bench nc] $(date -u +%H:%M:%S) start"
python src/eval/local_bench.py \
  --model-config configs/l1_signature/bench-bzr1v11-nc-lam70-mf14.yaml \
  --contexts Q_HepG2 --tag bzr1v11nc_Q > tmp/bench_v11nc_Q.log 2>&1
echo "[bench nc] rc=$? $(date -u +%H:%M:%S)"
python src/eval/bench_score.py --tags bzr1v11nc_Q --contexts Q_HepG2 > tmp/score_v11nc.log 2>&1
echo "[score nc] rc=$? $(date -u +%H:%M:%S)"
python scripts/official_alignment.py predict --tag bzr1v11nc_Q --family heur > tmp/pred_v11nc.json 2>tmp/pred_v11nc.err
echo "[pred nc] rc=$? $(date -u +%H:%M:%S)"
cat tmp/pred_v11nc.json
echo "[chain done] $(date -u +%H:%M:%S)"
