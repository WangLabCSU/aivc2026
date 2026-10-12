"""State ST 零样本输出 → 全局幅度换算层探针 (iter019).

背景: statest_se600m_e42 P_H1 nmae=51.1 已定性为幅度塌缩
(pred 跨靶 |Δ| median 1.14 vs ref 70.2, ~60x; 方向 cosine 多样性尚可 0.27 vs 0.07).
本探针在 log1p 空间对每靶 Δ 做全局标量缩放后重跑 cell-eval2, 扫 k 曲线:

    P'_target = mean_ntc_log + k·(P_target - mean_ntc_log)   (clip [0,15])
    X = expm1(P'); 行重归一到 TARGET_SUM (k=1 时恒等); raw = X/TARGET_SUM×lib

回答两问:
  1) 幅度校准能否把 nmae 从 51 拉回 λ 系水平 (~1.0)?
  2) pds/fid/reach/jac 随幅度的响应曲线 → 方向结构的价值定量锚点,
     供 C2 XGB g-head 的 y 目标与 C3 融合权重设计。

用法:
  /data/home/wsx/miniconda3/envs/aivc2026/bin/python scripts/probe_state_rescale.py \
      --context P_H1 --ks 1,3,10,30,60
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
sys.path.insert(0, str(ROOT))
from src.eval.local_bench import agg_mean, load_yaml, prepare_reference, run_celleval

_spec = importlib.util.spec_from_file_location(
    "state_infer_pred", ROOT / "scripts/state_infer_pred.py")
sip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sip)

LOG_CLIP = 15.0  # expm1 float32 溢出防护 (log1p 理论上限 ≈ 9.39)


def scaled_blocks(P: np.ndarray, tg: np.ndarray, ctrl: str, k: float,
                  mean_ntc: np.ndarray, Xntc: sp.csr_matrix, hpos: np.ndarray,
                  keep: np.ndarray, lib: np.ndarray, ntc_src: np.ndarray,
                  ctx_label: str):
    """缩放后的全 panel raw 块 (对齐 state_infer_pred.post_context, k=1 时恒等)."""
    Pk = P.copy()
    rows = np.where(tg != ctrl)[0]
    Pk[rows] = mean_ntc[None, :] + k * (Pk[rows] - mean_ntc[None, :])
    np.clip(Pk, 0.0, LOG_CLIP, out=Pk)
    X = np.expm1(Pk.astype(np.float64))  # 防溢出; 行重归一保质量守恒
    X /= np.maximum(X.sum(axis=1, keepdims=True), 1e-9) / sip.TARGET_SUM
    raw_hvg = X / sip.TARGET_SUM * lib[:, None]
    blocks, obs_all = [], []
    for p in pd.unique(tg):
        idx = np.where(tg == p)[0]
        full = np.asarray(Xntc[ntc_src[idx]].todense(), dtype=np.float32)
        full[:, hpos[keep]] = raw_hvg[idx][:, keep].astype(np.float32)
        np.clip(np.rint(full), 0, None, out=full)
        blocks.append(sp.csr_matrix(full))
        obs_all.append(pd.DataFrame({
            "context": ctx_label, "target_gene": p,
            "cell_id": [f"{ctx_label}_{p}_{i}" for i in range(len(idx))]}))
    return blocks, obs_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="P_H1")
    ap.add_argument("--ks", default="1,3,10,30,60")
    ap.add_argument("--sim", default=None)
    args = ap.parse_args()
    ks = [float(x) for x in args.ks.split(",") if x.strip()]
    ctx = args.context

    bench = load_yaml(sip.BENCH_CFG)
    ccfg = bench["contexts"][ctx]
    ctrl = str(ccfg["control"])
    ref = prepare_reference(ctx, ccfg, bench["n_cells_per_pert"], bench["n_ntc_ref"])
    panel, Xntc, targets = sip.read_ref(ref, "target_gene", ctrl)

    sim_path = Path(args.sim) if args.sim else sip.IN_DIR / f"bench_{ctx}_simulated.h5ad"
    sim = ad.read_h5ad(sim_path)
    P = np.asarray(sim.obsm["X_hvg"], dtype=np.float32)
    tg = sim.obs["target_gene"].astype(str).to_numpy()
    lib = sim.obs["lib_src"].to_numpy(dtype=np.float64)
    ntc_src = sim.obs["ntc_src"].to_numpy(dtype=int)
    ctrl_rows = np.where(tg == ctrl)[0]
    mean_ntc = P[ctrl_rows].mean(axis=0)
    print(f"[probe:{ctx}] P{P.shape} targets={len(targets)} ks={ks}", flush=True)

    hpos = panel.get_indexer(sip.load_hvg_names())
    keep = hpos >= 0
    out = ROOT / "outputs/metrics/local_bench/statest_rescale"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for k in ks:
        tag = f"{ctx}_k{int(k)}" if k == int(k) else f"{ctx}_k{k}"
        pp = out / f"{tag}_pred.h5ad"
        if not pp.exists():
            blocks, obs_all = scaled_blocks(P, tg, ctrl, k, mean_ntc, Xntc,
                                            hpos, keep, lib, ntc_src, ctx)
            pred = sip.assemble(blocks, obs_all, panel)
            del blocks, obs_all
            pred.write_h5ad(pp, compression="gzip")
            del pred
        agg = run_celleval(pp, ref, ccfg, out / f"{tag}_eval")
        m = agg_mean(agg)
        results[k] = m
        print(f"[k={k}] " + " ".join(f"{mm}={v:.4f}" for mm, v in m.items()), flush=True)
    overall = {str(k): float(np.nanmean(list(v.values()))) for k, v in results.items()}
    (out / f"summary_{ctx}.json").write_text(json.dumps(
        {"per_k": results, "overall_raw": overall}, indent=2, default=float))
    print(f"[probe:{ctx}] overall_raw per k: " +
          " ".join(f"k={v:.4f}" for v in overall.values()), flush=True)


if __name__ == "__main__":
    main()
