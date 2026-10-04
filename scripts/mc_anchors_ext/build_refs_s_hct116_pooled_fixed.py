"""S_HCT116 pooled 固定靶集 + 固定 pred 的纯 ref 重抽方差实验（归因确认终版）。

背景链: S_HCT116_deep（每 seed 重抽 100 靶, ±0.075）→ S_HCT116_fixed（固定 targets_s0 +
每 seed 重建 pred, ±0.092, 归因反转为结构性）。本实验再剥一层:
  - 靶集固定 = targets.json（浅版 bench 的 seed-0 100 靶, 即固定 pred 的扰动集）;
  - pred 固定 = outputs/metrics/local_bench/l11_lam1.5/S_HCT116_pred.h5ad
    （40,400×38,606; var 轴与 pooled 全量版完全同序, 扰动集=NTC+100 靶一致 → 无需重建,
    4 seed 共享同一 pred）;
  - 4 seed（0/11/12/13）仅重抽细胞: 每靶 min(400, pooled 可用量) + NTC 2000
    → ref_pfix_s{seed}.h5ad（~19.2k cells, 与任务背景 "~19.2k cells" 一致）。
→ overall5 std = 纯 ref 细胞重抽方差, 与 ±0.092（含 pred 重建）/±0.075（含靶集重抽）对照,
  判定 S_HCT116 判读线 <0.08 是否修订。

设计偏离（有意, 已记录）:
  - 任务原文 "每靶 30 细胞" 会让 ref 缩至 ~5k cells, 与任务自身 "~30-50k cells"/风险提示
    "~42k cells" 及背景 "~19.2k cells" 全部矛盾, 且非任何在案 bench 协议; 改用与固定 pred
    的产生协议（浅版 mc_error.build_ref）及 deep/fixed 节点一致的 request-400-cap-avail;
  - "每靶 ≥30 gate" 保留为资格门: targets.json 100 靶在 pooled 全量版中最少 30、中位 139.5,
    100/100 过门, 无靶被剔除（pred/real 扰动集一致性得以保持）;
  - 源 HCT116_pooled.h5ad 即 109 批全量版（.pooled_full.done: 3,409,169 cells × 38,606,
    n_batches=109; 任务提示 "796,139 cells=27 批版" 系旧部分构建, 现文件 139GB 为全量）。

源 139GB backed 只读所需行。用法: python scripts/mc_anchors_ext/build_refs_s_hct116_pooled_fixed.py
输出: data/interim/local_bench/S_HCT116/{ref_pfix_s*.h5ad,refs_pooled_fixed_meta.json}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
SRC = ROOT / "data/public/xatlas/HCT116_pooled.h5ad"
REF_DIR = ROOT / "data/interim/local_bench/S_HCT116"
PRED = ROOT / "outputs/metrics/local_bench/l11_lam1.5/S_HCT116_pred.h5ad"
PERT_COL, CONTROL = "target_gene", "Non-Targeting"
N_CELLS, N_NTC = 400, 2000
MIN_CELLS = 30
SEEDS = [0, 11, 12, 13]


def main() -> None:
    t0 = time.time()
    chosen = sorted(json.loads((REF_DIR / "targets.json").read_text()))
    assert len(chosen) == 100

    a = ad.read_h5ad(SRC, backed="r")
    tg = a.obs[PERT_COL].astype(str).to_numpy()
    var_names = np.asarray(a.var_names.astype(str))
    print(f"source {a.shape} opened backed in {time.time()-t0:.0f}s", flush=True)

    # ---- 固定 pred 一致性校验（轴 + 扰动集）; 通过则 4 seed 共享, 不重建 ----
    pv = ad.read_h5ad(PRED, backed="r")
    assert list(pv.var_names.astype(str)) == list(var_names), "pred var axis != pooled"
    pred_perts = set(pv.obs[PERT_COL].astype(str))
    assert pred_perts == set(chosen) | {CONTROL}, "pred pert set != targets.json + NTC"
    pv.file.close()
    print(f"fixed pred axis+pert set verified ({time.time()-t0:.0f}s)", flush=True)

    vc = pd.Series(tg).value_counts()
    short = {p: int(vc[p]) for p in chosen if vc[p] < MIN_CELLS}
    if short:
        raise RuntimeError(f"targets below {MIN_CELLS} cells in pooled: {short}")
    cnt = vc.reindex(chosen)
    print(f"gate >= {MIN_CELLS}: 100/100 pass (min={int(cnt.min())}, "
          f"median={float(cnt.median())}, >=400: {int((cnt >= N_CELLS).sum())})", flush=True)

    meta: dict = {}
    for seed in SEEDS:
        ref_path = REF_DIR / f"ref_pfix_s{seed}.h5ad"
        if ref_path.exists():
            print(f"[pfix] seed{seed}: reuse existing ref", flush=True)
            continue
        rng = np.random.default_rng(seed)
        keep_idx, per_pert = [], []
        for p in chosen + [CONTROL]:
            idx = np.where(tg == p)[0]
            take = rng.choice(idx, size=min(N_CELLS if p != CONTROL else N_NTC, len(idx)),
                              replace=False)
            per_pert.append(len(take))
            keep_idx.extend(take.tolist())
        keep_idx = sorted(keep_idx)
        ts = time.time()
        Xsub = sp.csr_matrix(a.X[keep_idx])          # backed 行子集（只读所需行）
        sub_obs = pd.DataFrame({"target_gene": tg[keep_idx]},
                               index=[f"r{i}" for i in range(len(keep_idx))])
        sub = ad.AnnData(X=Xsub, obs=sub_obs, var=pd.DataFrame(index=pd.Index(var_names)))
        sub.var_names = pd.Index(var_names)
        sub.var.index.name = None
        sub.write_h5ad(ref_path, compression=None)
        pp = np.array(per_pert[: len(chosen)])
        meta[f"seed{seed}"] = {"ref": str(ref_path),
                               "pred": str(PRED) + " (fixed, shared across seeds)",
                               "shape": list(sub.shape), "n_perts": len(chosen),
                               "median_cells_per_pert": float(np.median(pp)),
                               "min_cells_per_pert": int(pp.min()),
                               "n_perts_at_400": int((pp == N_CELLS).sum()),
                               "ntc_cells": int(per_pert[-1])}
        print(f"[pfix] seed{seed}: ref {sub.shape} written in {time.time()-ts:.0f}s "
              f"(total {time.time()-t0:.0f}s)", flush=True)
        del Xsub, sub
    a.file.close()

    meta["design"] = {
        "source": "HCT116_pooled.h5ad 109-batch full (3,409,169 x 38,606; .pooled_full.done)",
        "fixed_target_set": "targets.json (浅版 bench seed-0 100 靶 = 固定 pred 的扰动集); "
                            "gate >=30 cells: 100/100 pass (min 30, median 139.5, >=400: 7)",
        "n_cells_per_pert": "min(400, pooled availability) + NTC 2000 (request-400-cap-avail, "
                            "与浅版 bench/deep/fixed 节点协议一致)",
        "pred": "fixed shared l11_lam1.5/S_HCT116_pred.h5ad (axis+pert set verified), "
                "不随 seed 重建 → 与 S_HCT116_fixed 唯一差异 = pred 固定 + 靶集改回 bench 集",
        "pairwise_target_overlap": "100/100 (by construction)",
        "isolated_component": "纯 ref 细胞重抽（含 eval baseline/anchor 腿随 ref 重算）",
    }
    meta["seconds_total"] = round(time.time() - t0, 1)
    (REF_DIR / "refs_pooled_fixed_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
