#!/bin/bash
# 三源 ensemble 网格：build+eval 串行（GPU 单卡），每点 ~3.5min
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
run_one () {
  local tag=$1 w=$2
  if [ ! -f outputs/metrics/local_bench/$tag/Q_HepG2_scored.csv ]; then
    $PY scripts/ens3_scan.py run --w $w --lam 4.5 --tag $tag && \
    $PY scripts/ens3_scan.py eval --tag $tag
  fi
}
run_one ens3_c1 0.45,0.45,0.10,0
run_one ens3_c2 0.40,0.40,0.20,0
run_one ens3_c3 0.30,0.30,0.40,0
run_one ens3_b1 0.45,0.45,0,0.10
run_one ens3_b2 0.40,0.40,0,0.20
run_one ens3_m1 0.40,0.40,0.10,0.10
run_one ens3_m2 0.35,0.35,0.20,0.10
run_one ens3_s1 0.45,0.35,0.20,0
run_one ens3_s2 0.35,0.45,0.20,0
# 汇总
$PY - <<'PYEOF'
import pandas as pd, glob, os
rows=[]
for d in sorted(glob.glob("outputs/metrics/local_bench/ens3_*/")):
    csv=os.path.join(d,"Q_HepG2_scored.csv")
    if not os.path.exists(csv): continue
    s=pd.read_csv(csv)
    ov=float(s.loc[s.metric=="avg_score","from_replicate"].iloc[0]) if "avg_score" in set(s.metric) else float("nan")
    rows.append((os.path.basename(d.rstrip("/")),round(ov,4)))
for n,v in sorted(rows,key=lambda x:-x[1]): print(n,v)
PYEOF
