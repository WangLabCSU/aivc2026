"""S_HCT116 固定靶集深度多重复锚点参考构建（隔离靶集抽样方差）。

背景：S_HCT116_deep 节点（每 seed 重抽 100 靶, 两两重叠仅 6-13/100）overall5 std=±0.0746，
归因假设=靶集抽样。本实验固定靶集重跑同协议，若方差降至 RPE1 量级（±0.03）则假设成立。

设计（与 src/eval/rebuild_s_hct116_refs.py 深度协议逐项对齐，唯一变量=靶集）：
- 固定靶集 = targets_s0.json（深度版 seed-0 已选 100 靶, pooled 源中全部 >=400 细胞）。
  注: 任务原文指向的 targets.json（浅版 bench 靶集）在 pooled 源中位仅 139.5 cells/靶
  （仅 7/100 可达 400），无法支撑深度协议; 其"固定靶集"测量已存在 = 浅版节点 S_HCT116
  （overall5 ±0.0518）, 重跑无信息增益且与深度节点深度混杂, 故改用 targets_s0.json;
- 4 seed（0/11/12/13）各抽 每靶 400 细胞 + NTC 2000 → ref_fix_s{seed}.h5ad（42,000×38,606）;
- 每 seed 以自身 ref 的 NTC 块 bootstrap ⊕ 2^1.5Δ 重建 pred（rng 1000+seed, 同深度版绑定）
  → S_HCT116_pred_fix_s{seed}.h5ad（40,400×38,606）。
源 139GB backed 只读所需行。用法: python scripts/mc_anchors_ext/build_refs_s_hct116_fixed.py
输出: data/interim/local_bench/S_HCT116/{ref_fix_s*.h5ad,refs_fixed_meta.json}
      outputs/metrics/local_bench/l11_lam1.5/S_HCT116_pred_fix_s*.h5ad
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
sys.path.insert(0, str(ROOT))

SRC = ROOT / "data/public/xatlas/HCT116_pooled.h5ad"
REF_DIR = ROOT / "data/interim/local_bench/S_HCT116"
PRED_DIR = ROOT / "outputs/metrics/local_bench/l11_lam1.5"
MODEL_CFG = ROOT / "configs/l1_signature/ws-v1.2-sampled-lam15.yaml"
PERT_COL, CONTROL = "target_gene", "Non-Targeting"
N_CELLS, N_NTC = 400, 2000
SEEDS = [0, 11, 12, 13]


def main() -> None:
    from src.models.l1_signature.make_predictions import load_delta

    chosen = sorted(json.loads((REF_DIR / "targets_s0.json").read_text()))
    assert len(chosen) == 100
    mcfg = json.loads(json.dumps(__import__("yaml").safe_load(open(MODEL_CFG))))

    t0 = time.time()
    a = ad.read_h5ad(SRC, backed="r")
    tg = a.obs[PERT_COL].astype(str).to_numpy()
    var_names = np.asarray(a.var_names.astype(str))
    vc = pd.Series(tg).value_counts()
    short = {p: int(vc.get(p, 0)) for p in chosen if vc.get(p, 0) < N_CELLS}
    if short:
        raise RuntimeError(f"fixed targets below {N_CELLS} cells in pooled: {short}")
    print(f"source {a.shape} opened backed in {time.time()-t0:.0f}s; "
          f"100 fixed targets all >=400 cells (min={min(vc[p] for p in chosen)})", flush=True)

    # delta 只载一次（靶集固定, 4 seed 共享）
    panel = pd.Index(var_names)
    delta, rep = load_delta(mcfg, panel, chosen)
    dmap = delta.set_index(["target_gene", "gene"])["delta"]
    hits = len(set(chosen) & set(dmap.index.get_level_values(0)))
    print(f"delta loaded ({time.time()-t0:.0f}s), coverage {hits}/100", flush=True)

    meta = {}
    for seed in SEEDS:
        ref_path = REF_DIR / f"ref_fix_s{seed}.h5ad"
        pred_path = PRED_DIR / f"S_HCT116_pred_fix_s{seed}.h5ad"
        if ref_path.exists() and pred_path.exists():
            print(f"[fixed] seed{seed}: reuse existing ref+pred", flush=True)
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
        print(f"[fixed] seed{seed}: ref {sub.shape} written in {time.time()-ts:.0f}s", flush=True)

        # ---- pred 重建（深度版协议: 自身 ref NTC 块 bootstrap ⊕ λ·Δ, rng=1000+seed）----
        ts = time.time()
        mask = sub_obs[PERT_COL].to_numpy() == CONTROL
        Xntc = sp.csr_matrix(Xsub[mask])
        rng_p = np.random.default_rng(1000 + seed)
        blocks = []
        for p in [CONTROL] + chosen:
            M = Xntc[rng_p.choice(Xntc.shape[0], N_CELLS, replace=True)].copy()
            if p in dmap.index.get_level_values(0):
                d = dmap.loc[p].reindex(panel).to_numpy()
                f = np.power(2.0, 1.5 * np.nan_to_num(d, nan=0.0)).astype(np.float32)
                M = M.multiply(f.reshape(1, -1)).tocsr()
                M.data = np.clip(np.rint(M.data), 0, None).astype(np.float32)
                M.eliminate_zeros()
            blocks.append(M)
        X = sp.vstack(blocks, format="csr").astype(np.float32)
        X.eliminate_zeros()
        pred = ad.AnnData(X=X, obs=pd.DataFrame(
            {"target_gene": np.repeat([CONTROL] + chosen, N_CELLS), "context": "S_HCT116"},
            index=[f"p{i}_c{j}" for i in range(len(chosen) + 1) for j in range(N_CELLS)]))
        pred.var_names = panel
        pred.var.index.name = None
        pred.write_h5ad(pred_path, compression="gzip")
        pp = np.array(per_pert[: len(chosen)])
        meta[f"seed{seed}"] = {"ref": str(ref_path), "pred": str(pred_path),
                               "shape": list(sub.shape), "n_perts": len(chosen),
                               "median_cells_per_pert": float(np.median(pp)),
                               "min_cells_per_pert": int(pp.min()),
                               "n_perts_at_400": int((pp == N_CELLS).sum()),
                               "ntc_cells": int(per_pert[-1]), "delta_coverage": hits}
        print(f"[fixed] seed{seed}: pred {X.shape} written in {time.time()-ts:.0f}s "
              f"(total {time.time()-t0:.0f}s)", flush=True)
        del Xsub, sub, X, pred
    a.file.close()

    meta["design"] = {
        "fixed_target_set": "targets_s0.json (deep seed-0 chosen 100, all >=400 cells in pooled; "
                            "task 原文 targets.json 中位 139.5 cells/靶无法支撑 400/靶深度协议,"
                            "其固定靶集测量=浅版节点 S_HCT116)",
        "n_perts": 100, "n_cells_per_pert": N_CELLS, "n_ntc": N_NTC,
        "pairwise_target_overlap": "100/100 (by construction, vs deep node 6-13/100)",
        "pred_lambda": 1.5, "var_axis": "pooled 38,606 (ref==pred, cell-eval2 consistent)",
        "protocol_parity": "与 S_HCT116_deep 唯一差异=靶集固定(不再每 seed 重抽); "
                           "ref 抽样/pred NTC bootstrap(rng 1000+seed)/5 指标标度全同",
    }
    meta["seconds_total"] = round(time.time() - t0, 1)
    (REF_DIR / "refs_fixed_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
