"""E3: 为 P_H1/R_Jurkat 构建多重复锚点参考（seed=11,12,13）。
协议 = /tmp/mc_anchors/build_refs.py（Q_HepG2 版）逐字等价：固定 seed-0 targets.json 扰动集，
只变细胞/NTC 抽样 seed；每扰动 400 细胞 + NTC 2000。
用法: python build_refs.py --context P_H1
"""
import argparse, json, os, time
from pathlib import Path
import anndata as ad
import numpy as np

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
CTX = {
    "P_H1": dict(src="data/public/vcc2025/adata_Test.h5ad", pert_col="target_gene", control="non-targeting"),
    "R_Jurkat": dict(src="data/public/scperturb/Nadig2024_jurkat.h5ad", pert_col="gene", control="non-targeting"),
    "Q_HepG2": dict(src="data/public/scperturb/Nadig2024_hepg2.h5ad", pert_col="gene", control="non-targeting"),
    # 夜间任务: Replogle RPE1（磁盘实际文件名 Replogle2022_rpe1.h5ad；扰动列=gene；
    # 100 靶全在源中，min 可用量 7；协议同其他 ctx：400/扰动或可用量 + NTC 2000）
    "S_RPE1": dict(src="data/public/scperturb/Replogle2022_rpe1.h5ad", pert_col="gene", control="non-targeting"),
}
N_CELLS, N_NTC, SEEDS = 400, 2000, [11, 12, 13]
ap = argparse.ArgumentParser()
ap.add_argument("--context", required=True, choices=list(CTX))
args = ap.parse_args()
cfg = CTX[args.context]
# /tmp 仅 14G 禁用 → 默认重定向到 /data（env MC_ANCHORS_TMP 可覆盖以复用旧 /tmp 工件）
OUT = Path(os.environ.get("MC_ANCHORS_TMP", str(ROOT / "data/interim/mc_anchors_ext")))
OUT.mkdir(parents=True, exist_ok=True)
pert_set = json.loads((ROOT / f"data/interim/local_bench/{args.context}/targets.json").read_text())

t0 = time.time()
a = ad.read_h5ad(ROOT / cfg["src"])
print(f"source {a.shape} loaded in {time.time()-t0:.0f}s", flush=True)
tg = a.obs[cfg["pert_col"]].astype(str).to_numpy()

meta = {}
for seed in SEEDS:
    ref = OUT / f"{args.context}_seed{seed}.h5ad"
    if ref.exists():
        print(f"reuse {ref}", flush=True); continue
    rng = np.random.default_rng(seed)
    keep_idx, per_pert_counts = [], []
    for p in pert_set + [cfg["control"]]:
        idx = np.where(tg == p)[0]
        n = N_CELLS if p != cfg["control"] else N_NTC
        take = rng.choice(idx, size=min(n, len(idx)), replace=False)
        per_pert_counts.append(min(n, len(idx)))
        keep_idx.extend(take.tolist())
    sub = a[sorted(keep_idx)].copy()
    sub.obs = sub.obs[[cfg["pert_col"]]].rename(columns={cfg["pert_col"]: "target_gene"})
    sub.write_h5ad(ref, compression="gzip")
    pp = np.array(per_pert_counts[: len(pert_set)])
    meta[f"seed{seed}"] = {"path": str(ref), "shape": list(sub.shape),
        "median_cells_per_pert": float(np.median(pp)), "min_cells_per_pert": int(pp.min()),
        "max_cells_per_pert": int(pp.max()), "n_perts_truncated_at_400": int((pp >= 400).sum()),
        "ntc_cells": int(per_pert_counts[-1])}
    print(f"seed{seed}: {sub.shape} median={np.median(pp):.1f} min={pp.min()} max={pp.max()} "
          f"({time.time()-t0:.0f}s)", flush=True)
(OUT / f"{args.context}_refs_meta.json").write_text(json.dumps(meta, indent=2))
print("done", flush=True)
