"""双家族 Δ̂ 级 Ensemble：heuristic(dualsrc-deep) + fusion v3.5，α 扫描 Q_HepG2。"""
import sys, os
sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
os.chdir("/data/resource/wanglab/aivc/aivc2026")
os.environ["L15_CFG"] = "configs/l15_fusion/v35.yaml"

import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from pathlib import Path

# 1) bench Q 参考
ref = ad.read_h5ad('data/interim/local_bench/Q_HepG2/ref.h5ad')
print(f"ref: {ref.shape}, NTC: {(ref.obs.target_gene=='non-targeting').sum()}")

# 2) 启发式 Δ̂
gwps = pd.read_parquet('data/interim/signatures/gwps.parquet',
                       columns=['target_gene','gene','logfc'])
hct = pd.read_parquet('data/interim/signatures/hct116_all.parquet',
                      columns=['target_gene','gene','logfc'])
comb = pd.concat([gwps, hct])
heur = comb.groupby(['target_gene','gene'])['logfc'].mean().reset_index()
heur.columns = ['target_gene','gene','delta']

# 3) fusion Δ̂（bench Q 100 靶）——需要 fusion pred 或重建
# fusion v3.5 的 Q_HepG2 bench pred 如果存在则读取
fus_pred_candidates = list(Path('data/interim/l15_fusion/v35/').glob('pred_bench_*'))
print(f"fusion bench preds: {[str(f) for f in fus_pred_candidates]}")

# bench Q 100 靶
import json
bench_tgts = json.load(open('data/interim/local_bench/Q_HepG2/targets.json'))
print(f"bench targets: {len(bench_tgts)}")

# 4) 在 heur Δ̂ 中查 bench 靶覆盖
heur_tgts = set(heur['target_gene'].unique())
bench_covered = set(bench_tgts) & heur_tgts
print(f"bench targets in heur: {len(bench_covered)}/{len(bench_tgts)}")

# 5) fusion Δ̂ 在 bench 靶上的覆盖
# fusion v3.5 的 bench pred 可能在 data/interim/l15_fusion/v35/feat_bench_hepg2.npz
feat_bench = Path('data/interim/l15_fusion/v35/feat_bench_hepg2.npz')
print(f"feat_bench exists: {feat_bench.exists()}")
if feat_bench.exists():
    z = np.load(feat_bench, allow_pickle=False)
    print(f"feat keys: {list(z.keys())[:5]}")
    if 'target' in z:
        ftgts = set(z['target'].astype(str).tolist())
        print(f"fusion bench targets: {len(ftgts)}, overlap with bench: {len(ftgts & set(bench_tgts))}")
