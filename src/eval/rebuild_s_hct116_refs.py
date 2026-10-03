"""S_HCT116 深度参考重建（多重复锚点补齐·深度版）：建 ref + 建 pred 一体化，保证轴一致。

背景：S_HCT116 浅版多重复锚点（mc_anchors.json/S_HCT116）的 ref 方差 ±0.0403 全场最大
——根因是固定 seed-0 靶集在 pooled 源里中位只有 ~140 cells/靶（min30 gate），锚点腿
（5 个 half-split）与 baseline 腿的方差都大。深度版改为:

  - gate: pooled 源中每扰动 >=400 细胞的靶（1,062 个）里每 seed 随机抽 100 个；
  - 每靶取 400 细胞 + NTC('Non-Targeting') 2000 → ref_s{seed}.h5ad（无压缩写快）；
  - 每 seed 用同一 ref 的 var 轴（= pooled var, 38,606）重建候选 pred
    （l11 λ1.5 构建逻辑: ref NTC bootstrap 采样 ⊕ 2^(1.5·Δ)）——cell-eval2 要求
    pred/real 基因轴与扰动集合一致，故 pred 必须随 ref 每 seed 重建。

源 139GB（~3.4M cells × 38,606）anndata backed 只读所需行（同
mc_anchors_ext/build_refs_s_hct116.py 协议），避免 full read OOM。

用法: python src/eval/rebuild_s_hct116_refs.py [--seeds 0,11,12,13]
输出: data/interim/local_bench/S_HCT116/ref_s{seed}.h5ad + targets_s{seed}.json
      outputs/metrics/local_bench/l11_lam1.5/S_HCT116_pred_s{seed}.h5ad
      data/interim/local_bench/S_HCT116/refs_deep_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "data/public/xatlas/HCT116_pooled.h5ad"
REF_DIR = ROOT / "data/interim/local_bench/S_HCT116"
PRED_DIR = ROOT / "outputs/metrics/local_bench/l11_lam1.5"
MODEL_CFG = ROOT / "configs/l1_signature/ws-v1.2-sampled-lam15.yaml"
PERT_COL, CONTROL = "target_gene", "Non-Targeting"
N_CELLS, N_NTC, N_PERTS = 400, 2000, 100
MIN_CELLS = 400


def build_ref_and_pred(a: ad.AnnData, tg: np.ndarray, var_names: np.ndarray,
                       seed: int, lam: float) -> tuple[Path, Path, dict]:
    from src.models.l1_signature.make_predictions import load_delta

    ref_path = REF_DIR / f"ref_s{seed}.h5ad"
    pred_path = PRED_DIR / f"S_HCT116_pred_s{seed}.h5ad"
    if ref_path.exists() and pred_path.exists():
        print(f"[deep] seed{seed}: reuse existing ref+pred", flush=True)
        meta = json.loads((REF_DIR / f"targets_s{seed}.json").read_text())
        return ref_path, pred_path, meta

    rng = np.random.default_rng(seed)
    vc = pd.Series(tg).value_counts()
    eligible = sorted(vc[(vc.index != CONTROL) & (vc >= MIN_CELLS)].index)
    chosen = sorted(rng.choice(eligible, size=min(N_PERTS, len(eligible)), replace=False))
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
    sub.write_h5ad(ref_path, compression=None)   # 无压缩写快
    print(f"[deep] seed{seed}: ref {sub.shape} written in {time.time()-ts:.0f}s", flush=True)

    # ---- pred 重建（local_bench 采样逻辑, 同一 ref 的 NTC 块 ⊕ λ·Δ, 轴=ref var）----
    ts = time.time()
    mcfg = json.loads(json.dumps(__import__("yaml").safe_load(open(MODEL_CFG))))
    panel = pd.Index(var_names)
    delta, rep = load_delta(mcfg, panel, chosen)
    dmap = delta.set_index(["target_gene", "gene"])["delta"]
    hits = len(set(chosen) & set(dmap.index.get_level_values(0)))
    mask = sub_obs[PERT_COL].to_numpy() == CONTROL
    Xntc = sp.csr_matrix(Xsub[mask])
    rng_p = np.random.default_rng(1000 + seed)   # pred bootstrap 与 ref seed 绑定
    blocks = []
    for p in [CONTROL] + chosen:
        M = Xntc[rng_p.choice(Xntc.shape[0], N_CELLS, replace=True)].copy()
        if p in dmap.index.get_level_values(0):
            d = dmap.loc[p].reindex(panel).to_numpy()
            f = np.power(2.0, lam * np.nan_to_num(d, nan=0.0)).astype(np.float32)
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
    print(f"[deep] seed{seed}: pred {pred.shape} written in {time.time()-ts:.0f}s "
          f"(delta coverage {hits}/{len(chosen)})", flush=True)

    (REF_DIR / f"targets_s{seed}.json").write_text(json.dumps(chosen))
    pp = np.array(per_pert[: len(chosen)])
    meta = {"ref": str(ref_path), "pred": str(pred_path),
            "shape": list(sub.shape), "n_perts": len(chosen),
            "median_cells_per_pert": float(np.median(pp)), "min_cells_per_pert": int(pp.min()),
            "n_perts_at_400": int((pp == N_CELLS).sum()), "ntc_cells": int(per_pert[-1]),
            "delta_coverage": hits}
    return ref_path, pred_path, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,11,12,13")
    ap.add_argument("--lam", type=float, default=1.5)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    REF_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    a = ad.read_h5ad(SRC, backed="r")
    tg = a.obs[PERT_COL].astype(str).to_numpy()
    var_names = np.asarray(a.var_names.astype(str))
    print(f"[deep] source {a.shape} opened backed in {time.time()-t0:.0f}s", flush=True)

    meta: dict = {}
    for seed in seeds:
        _, _, m = build_ref_and_pred(a, tg, var_names, seed, args.lam)
        meta[f"seed{seed}"] = m
        print(f"[deep] seed{seed}: {json.dumps(m)}", flush=True)
    a.file.close()

    # 靶集重叠矩阵（设计说明: 每 seed 独立抽 100 靶, 非浅版的固定靶集协议）
    sets = {s: set(json.loads((REF_DIR / f"targets_s{s}.json").read_text())) for s in seeds}
    meta["design"] = {
        "gate_min_cells_per_pert": MIN_CELLS, "eligible_perts": 1062,
        "n_perts_per_ref": N_PERTS, "n_cells_per_pert": N_CELLS, "n_ntc": N_NTC,
        "pert_set": "per-seed random 100 of >=400-cell perts (pred rebuilt per seed; "
                    "differs from shallow node's fixed seed-0 target set)",
        "pred_lambda": args.lam, "var_axis": "pooled 38,606 (ref==pred, cell-eval2 consistent)",
        "pairwise_target_overlap": {f"{x}-{y}": len(sets[x] & sets[y])
                                    for i, x in enumerate(seeds) for y in seeds[i + 1:]},
    }
    meta["seconds_total"] = round(time.time() - t0, 1)
    (REF_DIR / "refs_deep_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[deep] ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
