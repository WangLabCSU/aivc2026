"""E5: S_HCT116 多重复锚点参考构建（seed=11,12,13）——139GB pooled 源的 backed 行子集版。

协议 = mc_anchors_ext/build_refs.py + local_bench.prepare_reference：
- 固定 seed-0 targets.json 的 100 靶（cell-eval2 要求 pred/real 扰动集合一致）；
- 只变细胞/NTC 抽样 seed；每扰动 400 细胞 + NTC 2000（不足则全取）；
- 源 data/public/xatlas/HCT116_pooled.h5ad（~280GB csr）只读所需行，避免 OOM；
- obs 只留 target_gene（str，去空类目），var 对齐源。

用法: python scripts/mc_anchors_ext/build_refs_s_hct116.py
输出: $MC_ANCHORS_TMP(默认 data/interim/mc_anchors_ext)/S_HCT116_seed{11,12,13}.h5ad + refs_meta.json
"""
import json
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
SRC = ROOT / "data/public/xatlas/HCT116_pooled.h5ad"
PERT_COL, CONTROL = "target_gene", "Non-Targeting"
N_CELLS, N_NTC, SEEDS = 400, 2000, [11, 12, 13]
OUT = Path(os.environ.get("MC_ANCHORS_TMP", str(ROOT / "data/interim/mc_anchors_ext")))
OUT.mkdir(parents=True, exist_ok=True)
pert_set = json.loads((ROOT / "data/interim/local_bench/S_HCT116/targets.json").read_text())

t0 = time.time()
a = ad.read_h5ad(SRC, backed="r")
tg = a.obs[PERT_COL].astype(str).to_numpy()
print(f"source {a.shape} opened (backed) in {time.time()-t0:.0f}s", flush=True)
counts = pd_count = None
import pandas as pd
vc = pd.Series(tg).value_counts()
dropped = [p for p in pert_set if vc.get(p, 0) < 30]
if dropped:
    raise RuntimeError(f"seed-0 perts below min_cells 30 in pooled source: {dropped}")

var_names = np.asarray(a.var_names.astype(str))
meta = {}
for seed in SEEDS:
    ref = OUT / f"S_HCT116_seed{seed}.h5ad"
    if ref.exists():
        print(f"reuse {ref}", flush=True)
        continue
    rng = np.random.default_rng(seed)
    keep_idx, per_pert_counts = [], []
    for p in pert_set + [CONTROL]:
        idx = np.where(tg == p)[0]
        n = N_CELLS if p != CONTROL else N_NTC
        take = rng.choice(idx, size=min(n, len(idx)), replace=False)
        per_pert_counts.append(len(take))
        keep_idx.extend(take.tolist())
    keep_idx = sorted(keep_idx)
    ts = time.time()
    Xsub = sp.csr_matrix(a.X[keep_idx])   # backed 行子集
    sub_obs = pd.DataFrame({"target_gene": tg[keep_idx]},
                           index=[f"r{i}" for i in range(len(keep_idx))])
    sub = ad.AnnData(X=Xsub, obs=sub_obs, var=pd.DataFrame(index=pd.Index(var_names)))
    sub.var_names = pd.Index(var_names)
    sub.var.index.name = None
    sub.write_h5ad(ref, compression="gzip")
    pp = np.array(per_pert_counts[: len(pert_set)])
    meta[f"seed{seed}"] = {"path": str(ref), "shape": list(sub.shape),
                           "median_cells_per_pert": float(np.median(pp)),
                           "min_cells_per_pert": int(pp.min()),
                           "max_cells_per_pert": int(pp.max()),
                           "n_perts_truncated_at_400": int((pp >= 400).sum()),
                           "ntc_cells": int(per_pert_counts[-1])}
    print(f"seed{seed}: {sub.shape} median={np.median(pp):.1f} min={pp.min()} max={pp.max()} "
          f"({time.time()-ts:.0f}s, total {time.time()-t0:.0f}s)", flush=True)
    del Xsub, sub
a.file.close()
(OUT / "S_HCT116_refs_meta.json").write_text(json.dumps(meta, indent=2))
print("done", flush=True)
