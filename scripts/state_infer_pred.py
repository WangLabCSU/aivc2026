"""State ST (SE-600M 条件化重训) 推理 → 官方提交 / 本地 bench 评测.

三段式: prep(构造推理输入 h5ad) → infer(state tx infer) → post(反归一化 + 组装 + 评测).

用法:
  # 官方提交 (context A/B/C → 301 perts×400 cells×3 ctx = 361,200 行 18,533 raw counts)
  python scripts/state_infer_pred.py --mode challenge --out-stem 20261013_statest_se600m

  # 本地 bench (P_H1/Q_HepG2/R_Jurkat → cell-eval2 六指标)
  python scripts/state_infer_pred.py --mode bench --tag statest_se600m_e42

  # 分段运行 (prep / infer / post)
  python scripts/state_infer_pred.py --mode bench --stage prep

通路确证 (iter018, 勿再挖):
  - 输入归一化 normalize_total(target_sum=11940)+log1p (rpe1_prep expm1 行和=11940 精确)
  - 特征列 = var_dims['gene_names'] 8192 (rpe1 HVG 序), 按 panel 名对齐, 缺失置零
  - obs 不注入 batch/celltype 列 → infer 自动 '__ALL__' 单组 + batch 编码器禁用
  - pert 向量: pert_onehot_map (SE-600M 5120 维), 300 挑战靶 100% 覆盖
  - 输出 <input>_simulated.h5ad 的 obsm['X_hvg'] = 8192 log1p 预测 (全行覆写含对照)
  - 反归一化: raw = expm1(pred)/11940 × 源NTC行和; 非 HVG 基因用源 NTC raw 值填充
  - 挑战靶列表必须自 data/raw/controls/pert_counts.csv 注入:
    context_A/B/C.h5ad 是纯 NTC 池 (obs 无扰动靶), 从 obs 推导 targets 恒为空
  - cell-eval2 要求 pred 的 pert 集合与 ref 一致 (含对照): pred_perts = [control] + targets
  - var_dims.pkl 为自产可信训练产物
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
ENV_BIN = Path("/data/home/wsx/miniconda3/envs/aivc2026/bin")
MDIR = ROOT / "data/interim/state_retrain/run_v2/state_se600m_gwps_rpe1"
IN_DIR = ROOT / "data/interim/state_retrain/infer_in"
SUB_DIR = ROOT / "outputs/submissions"
CTRLS = ROOT / "data/raw/controls"
BENCH_CFG = ROOT / "configs/local_bench/contexts.yaml"
CTRL = "non-targeting"
TARGET_SUM = 11940.0
N_CELLS = 400
PREP_SEED = 0
INFER_SEED = 42


def load_hvg_names() -> pd.Index:
    """var_dims.pkl (自产可信训练产物) → 8192 HVG 名 (模型特征列序)."""
    with open(MDIR / "var_dims.pkl", "rb") as f:
        vd = pickle.load(f)
    return pd.Index([str(g) for g in vd["gene_names"]])


def norm_log1p(Xntc: sp.csr_matrix) -> sp.csr_matrix:
    """raw counts → log1p(normalize_total(target_sum=11940)) (对齐 rpe1_prep 训练输入)."""
    libs = np.asarray(Xntc.sum(axis=1)).ravel()
    w = TARGET_SUM / np.maximum(libs, 1.0)
    Xn = Xntc.astype(np.float32).multiply(w[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data)
    return Xn


def read_ref(path: Path, pert_col: str, ctrl_label: str):
    """读 adata → (panel 名, 对照池 raw, 靶列表). panel 在 del 前取出 (防 use-after-del)."""
    a = ad.read_h5ad(path)
    panel = pd.Index(a.var_names.astype(str))
    tg = a.obs[pert_col].astype(str).to_numpy()
    Xntc = sp.csr_matrix(a[tg == ctrl_label].X, dtype=np.float64)
    targets = sorted(set(tg.tolist()) - {ctrl_label})
    del a
    return panel, Xntc, targets


def build_input(panel: pd.Index, Xntc: sp.csr_matrix, targets: list, ctrl_label: str,
                ctx_label: str, in_path: Path, sim_path: Path) -> None:
    """对照池 raw → 推理输入 h5ad: 每 pert 400 虚拟克隆行, obsm['X_hvg'] 8192 维."""
    gene_names = load_hvg_names()
    hpos = panel.get_indexer(gene_names)
    keep = hpos >= 0
    print(f"[prep:{ctx_label}] hvg 覆盖 {int(keep.sum())}/{len(gene_names)} targets={len(targets)}", flush=True)

    libs = np.asarray(Xntc.sum(axis=1)).ravel()
    Xn = norm_log1p(Xntc)
    Xn_csc = Xn.tocsc()
    Xbase = np.zeros((Xn.shape[0], len(gene_names)), dtype=np.float32)
    Xbase[:, keep] = np.asarray(Xn_csc[:, hpos[keep]].todense(), dtype=np.float32)

    rng = np.random.default_rng(PREP_SEED)
    n_ntc = Xbase.shape[0]
    blocks, recs = [], []
    for p in [ctrl_label] + list(targets):
        src = rng.choice(n_ntc, N_CELLS, replace=True)
        blocks.append(Xbase[src])
        recs.append(pd.DataFrame({
            "target_gene": p, "context": ctx_label,
            "cell_id": [f"{ctx_label}_{p}_{i}" for i in range(N_CELLS)],
            "lib_src": libs[src], "ntc_src": src}))
    X_in = np.vstack(blocks).astype(np.float32)
    obs = pd.concat(recs, ignore_index=True).set_index("cell_id")
    if in_path.exists():
        in_path.unlink()
    A = ad.AnnData(X=sp.csr_matrix(X_in), obs=obs,
                   var=pd.DataFrame(index=gene_names))
    A.obsm["X_hvg"] = X_in
    in_path.parent.mkdir(parents=True, exist_ok=True)
    A.write_h5ad(in_path, compression="gzip")
    print(f"[prep:{ctx_label}] {in_path} shape={X_in.shape}", flush=True)
    _ = sim_path  # 由 run_infer 按 <in>_simulated.h5ad 约定使用


def run_infer(in_path: Path, sim_path: Path, checkpoint: str, ctrl_label: str = CTRL) -> None:
    if sim_path.exists():
        print(f"[infer] skip, exists: {sim_path}", flush=True)
        return
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    cmd = [str(ENV_BIN / "state"), "tx", "infer",
           "--model-dir", str(MDIR), "--adata", str(in_path),
           "--pert-col", "target_gene", "--control-pert", ctrl_label,
           "--min-cells", str(N_CELLS), "--max-cells", str(N_CELLS),
           "--seed", str(INFER_SEED), "--embed-key", "X_hvg",
           "--output", str(sim_path), "--quiet"]
    print("[infer] " + " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"[infer] done in {time.time()-t0:.0f}s -> {sim_path}", flush=True)


def post_context(ctx_label: str, sim_path: Path, panel: pd.Index,
                 Xntc: sp.csr_matrix, blocks: list, obs_all: list) -> None:
    """sim 的 obsm['X_hvg'] (8192 log1p) → 反归一化 + 非 HVG 基因源 NTC 基底填充 → 全 panel raw 块."""
    hpos = panel.get_indexer(load_hvg_names())
    keep = hpos >= 0
    sim = ad.read_h5ad(sim_path)
    P = np.asarray(sim.obsm["X_hvg"], dtype=np.float32)
    lib = sim.obs["lib_src"].to_numpy(dtype=np.float64)
    ntc = sim.obs["ntc_src"].to_numpy(dtype=int)
    tg = sim.obs["target_gene"].astype(str).to_numpy()
    raw_hvg = np.expm1(P) / TARGET_SUM * lib[:, None]
    for p in pd.unique(tg):
        idx = np.where(tg == p)[0]
        full = np.asarray(Xntc[ntc[idx]].todense(), dtype=np.float32)
        full[:, hpos[keep]] = raw_hvg[idx][:, keep]
        np.clip(np.rint(full), 0, None, out=full)
        blocks.append(sp.csr_matrix(full))
        obs_all.append(pd.DataFrame({
            "context": ctx_label, "target_gene": p,
            "cell_id": [f"{ctx_label}_{p}_{i}" for i in range(len(idx))]}))
    del sim, P, raw_hvg
    print(f"[post:{ctx_label}] blocks={len(blocks)} rows={sum(len(r) for r in obs_all)}", flush=True)


def assemble(blocks: list, obs_all: list, panel: pd.Index) -> ad.AnnData:
    """vstack 块 → AnnData; var_names=panel, index.name=None (提交/评测范本)."""
    X = sp.vstack(blocks, format="csr").astype(np.float32)
    X.eliminate_zeros()
    obs = pd.concat(obs_all, ignore_index=True).set_index("cell_id")
    pred = ad.AnnData(X=X, obs=obs)
    pred.var_names = panel.to_numpy()
    pred.var.index.name = None
    return pred


def run_challenge(stem: str, checkpoint: str, stage: str) -> None:
    """官方提交: context A/B/C (纯 NTC 池) × 300 靶 (pert_counts.csv) → 361,200 行."""
    targets = pd.read_csv(CTRLS / "pert_counts.csv").iloc[:, 0].astype(str).tolist()
    if len(targets) != 300:
        raise ValueError(f"pert_counts.csv 靶数异常: {len(targets)} (期望 300)")
    blocks, obs_all = [], []
    panel = None
    for cx in ["A", "B", "C"]:
        cpath = CTRLS / f"context_{cx}.h5ad"
        in_path = IN_DIR / f"chal_{cx}.h5ad"
        sim_path = IN_DIR / f"chal_{cx}_simulated.h5ad"
        if stage in ("prep", "all"):
            p, Xntc, _ = read_ref(cpath, "target_gene", CTRL)
            build_input(p, Xntc, targets, CTRL, cx, in_path, sim_path)
            del Xntc
        if stage in ("infer", "all"):
            run_infer(in_path, sim_path, checkpoint, CTRL)
        if stage in ("post", "all"):
            p, Xntc, _ = read_ref(cpath, "target_gene", CTRL)
            if panel is None:
                panel = p
            elif not panel.equals(p):
                raise RuntimeError(f"context_{cx} panel 与 A 不一致")
            post_context(cx, sim_path, p, Xntc, blocks, obs_all)
            del Xntc
    if stage in ("post", "all") and obs_all:
        pred = assemble(blocks, obs_all, panel)
        out = SUB_DIR / f"{stem}.h5ad"
        out.parent.mkdir(parents=True, exist_ok=True)
        pred.write_h5ad(out, compression="gzip")
        print(f"[submission] {out} shape={pred.shape}", flush=True)


def run_bench(contexts: list, tag: str, checkpoint: str, stage: str) -> None:
    """本地 bench: prepare_reference 缓存 ref → 同管线 → cell-eval2 六指标."""
    sys.path.insert(0, str(ROOT))
    from src.eval.local_bench import (METRICS, agg_mean, load_yaml,
                                      prepare_reference, run_celleval)
    bench = load_yaml(BENCH_CFG)
    out = ROOT / "outputs/metrics/local_bench" / tag
    out.mkdir(parents=True, exist_ok=True)
    per_ctx = {}
    for ctx in contexts:
        ccfg = bench["contexts"][ctx]
        ctrl = str(ccfg["control"])
        ref = prepare_reference(ctx, ccfg, bench["n_cells_per_pert"], bench["n_ntc_ref"])
        in_path = IN_DIR / f"bench_{ctx}.h5ad"
        sim_path = IN_DIR / f"bench_{ctx}_simulated.h5ad"
        if stage in ("prep", "all"):
            panel, Xntc, targets = read_ref(ref, "target_gene", ctrl)
            build_input(panel, Xntc, targets, ctrl, ctx, in_path, sim_path)
            del Xntc
        if stage in ("infer", "all"):
            run_infer(in_path, sim_path, checkpoint, ctrl)
        if stage in ("post", "all"):
            panel, Xntc, _ = read_ref(ref, "target_gene", ctrl)
            blocks, obs_rec = [], []
            post_context(ctx, sim_path, panel, Xntc, blocks, obs_rec)
            del Xntc
            pred = assemble(blocks, obs_rec, panel)
            pp = out / f"{ctx}_pred.h5ad"
            pred.write_h5ad(pp, compression="gzip")
            agg = run_celleval(pp, ref, ccfg, out / f"{ctx}_eval")
            per_ctx[ctx] = agg_mean(agg)
            print(f"[{ctx}] " + " ".join(f"{m}={v:.4f}" for m, v in per_ctx[ctx].items()), flush=True)
    if per_ctx:
        overall = float(np.nanmean([np.nanmean(list(v.values())) for v in per_ctx.values()]))
        (out / "summary.json").write_text(json.dumps(
            {"per_context": per_ctx, "overall_raw": overall}, indent=2))
        print(f"[bench] overall_raw={overall:.4f} -> {out / 'summary.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="State ST 推理 → 官方提交 / 本地 bench")
    ap.add_argument("--mode", choices=["challenge", "bench"], required=True)
    ap.add_argument("--stage", choices=["all", "prep", "infer", "post"], default="all")
    ap.add_argument("--out-stem", default=None, help="challenge 模式提交文件名 (无 .h5ad)")
    ap.add_argument("--tag", default="statest_se600m", help="bench 模式输出子目录")
    ap.add_argument("--checkpoint", default=str(MDIR / "checkpoints" / "final.ckpt"))
    ap.add_argument("--contexts", default="P_H1,Q_HepG2,R_Jurkat", help="bench 上下文, 逗号分隔")
    args = ap.parse_args()
    if args.mode == "challenge":
        stem = args.out_stem or time.strftime("%Y%m%d_statest")
        run_challenge(stem, args.checkpoint, args.stage)
    else:
        run_bench([c.strip() for c in args.contexts.split(",") if c.strip()],
                  args.tag, args.checkpoint, args.stage)


if __name__ == "__main__":
    main()
