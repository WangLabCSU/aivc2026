#!/bin/bash
# v1.1 vs v1.0 skip 载体成对 λ 扫描（当前特征重算版）
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026
export BIZNET_CFG=configs/bizard/bizardnet_v11.yaml
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python
# v1.1 bench predict（full 输出, 用于 gate 诊断）
$PY src/models/bizard/bizardnet.py predict --src bench_hepg2 --arm loco
# v1.1 skip Δ̂ 重算
$PY - <<'PYEOF'
import numpy as np, pandas as pd, torch
ds="data/interim/bizard/bizardnet/bench_hepg2/"
z=np.load(ds+"order.npz", allow_pickle=True)
genF=np.load(ds+"genF.npy")
ck=torch.load("data/interim/bizard/bizardnet11/bizardnet_loco.pt", map_location="cpu", weights_only=False)
W=ck["state"]["skip.weight"].cpu().numpy().ravel(); b=float(ck["state"]["skip.bias"].item())
pred=(genF@W+b).astype(np.float32)
orig=np.empty(len(pred),np.float32); orig[z["ord"]]=pred
pd.DataFrame({"target_gene":z["target"].astype(str),"gene":z["gene"].astype(str),"delta":orig}) \
  .to_parquet("data/interim/bizard/bizardnet11/skipv11_bench_hepg2.parquet")
print("skipv11 sd", round(float(orig.std()),4))
PYEOF
# 成对 λ 扫描: v1.0(重算) 与 v1.1
for lam in 12 16 18 20; do
  $PY src/models/bizard/bench_bird.py --tag skipv10r_lam$lam --lam $lam --contexts Q_HepG2 \
    --delta-dir data/interim/bizard/bizardnet11 --delta-pattern "skipv10_bench_hepg2.parquet"
  $PY src/models/bizard/bench_bird.py --tag skipv11_lam$lam --lam $lam --contexts Q_HepG2 \
    --delta-dir data/interim/bizard/bizardnet11 --delta-pattern "skipv11_bench_hepg2.parquet"
done
# 汇总
$PY src/eval/bench_score.py --tags skipv10r_lam12,skipv11_lam12,skipv10r_lam16,skipv11_lam16,skipv10r_lam18,skipv11_lam18,skipv10r_lam20,skipv11_lam20 --contexts Q_HepG2 --out data/interim/bizard/bizardnet11/anchor_v11_sweep.json 2>&1 | grep -E "overall|multi-ref"
