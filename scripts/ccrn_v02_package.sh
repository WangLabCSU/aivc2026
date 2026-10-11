#!/bin/bash
# CCRN v0.2 (PBT winner) 完整打包: full 臂 → A/B/C predict → 中心化 → build → prep
set -e
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
cd /data/resource/wanglab/aivc/aivc2026
PY=/data/home/wsx/miniconda3/envs/aivc2026/bin/python

# 1) 取 PBT 最优 hp → config（workdir=ccrn_v02, 缓存 symlink）
W=$($PY - <<'PYEOF'
import json
st=json.load(open("data/interim/bizard/ccrn_pbt/pbt_state.json"))
best=max(st["workers"].items(), key=lambda kv: kv[1]["reward"] or -9)
print(best[0])
import yaml, os
from pathlib import Path
hp=best[1]["hp"]
base=yaml.safe_load(open("configs/bizard/ccrn.yaml"))
base["workdir"]="data/interim/bizard/ccrn_v02"
base["model"]["hidden"]=hp["hidden"]; base["model"]["dropout"]=hp["dropout"]
base["train"]["lr"]=hp["lr"]; base["train"]["seed"]=hp["seed"]
Path("configs/bizard/ccrn_v02.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
d=Path("data/interim/bizard/ccrn_v02"); d.mkdir(parents=True, exist_ok=True)
for f in ["X_train_ccrn.npy","X_bench_hepg2_ccrn.npy","meta_bench_hepg2.npz","train_meta.npz",
          "fingerprints.npy","X_submit_A_ccrn.npy","X_submit_B_ccrn.npy","X_submit_C_ccrn.npy",
          "meta_submit_A.npz","meta_submit_B.npz","meta_submit_C.npz"]:
    src=Path("data/interim/bizard/ccrn")/f; dst=d/f
    if src.exists() and not dst.exists(): dst.symlink_to(os.path.relpath(src, d))
print("best hp:", hp, file=os.sys.stderr)
PYEOF
)
export CCRN_CFG=configs/bizard/ccrn_v02.yaml
echo "worker $W config ready"

# 2) full 臂训练 + A/B/C 预测
$PY src/models/bizard/ccrn.py train --arm full
for cx in A B C; do $PY src/models/bizard/ccrn.py predict --src submit_$cx --arm full; done

# 3) per-context 中心化 → delta_submit
$PY - <<'PYEOF'
import pandas as pd
for cx in ["A","B","C"]:
    p=pd.read_parquet(f"data/interim/bizard/ccrn_v02/pred_ccrn1_full_submit_{cx}.parquet")
    p["delta"]=p["delta"]-p["delta"].mean()
    p.to_parquet(f"data/interim/bizard/ccrn_v02/delta_submit_{cx}.parquet")
    print(cx, "centered sd", round(float(p.delta.std()),4))
PYEOF

# 4) λ 标定 + build + prep（λ 由 bench 峰 × sd 比给出，脚本内打印建议值）
$PY - <<'PYEOF'
import pandas as pd, numpy as np, json
sds={}
for cx in ["A","B","C"]:
    d=pd.read_parquet(f"data/interim/bizard/ccrn_v02/delta_submit_{cx}.parquet")
    sds[cx]=float(d.delta.std())
print("submit sds:", {k: round(v,4) for k,v in sds.items()})
PYEOF
