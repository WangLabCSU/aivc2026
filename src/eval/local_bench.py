"""本地独立评测面板 local-bench：模拟 val A/B/C 的多上下文持续迭代基准。

三个伪上下文（P_H1/Q_HepG2/R_Jurkat）各含 NTC + 靶基因列表（模型可见）与真值
（评测专用）。每个候选模型在此出全量六指标，聚合口径对齐官方（六指标无权平均
→ 上下文无权平均）。通过后才消耗线上额度。

用法:
  python src/eval/local_bench.py --model-config configs/l1_signature/ws-v1.1-sampled.yaml \
      [--contexts P_H1,Q_HepG2,R_Jurkat] [--tag my_variant] [--anchor]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
ENV_BIN = Path("/data/home/wsx/miniconda3/envs/aivc2026/bin")
BENCH_DIR = ROOT / "data/interim/local_bench"
OUT_DIR = ROOT / "outputs/metrics/local_bench"
METRICS = ["pds_cosine", "expr_mse_unbiased", "de_wilcoxon_lfc_nmae",
           "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
           "de_wilcoxon_sig_jaccard"]


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_reference(ctx: str, ccfg: dict, n_cells: int, n_ntc: int) -> Path:
    """子采样真值参考（每扰动 n_cells 细胞 + n_ntc 对照），缓存一次。

    大源文件（HCT116 全深度 pooled ~280GB csr）用 backed 行子集读取，
    避免 full read 的 ~300GB RSS 撞 cgroup OOM。
    """
    out = BENCH_DIR / ctx
    ref = out / "ref.h5ad"
    if ref.exists():
        return ref
    out.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(ROOT / ccfg["h5ad"], backed="r")
    tg = a.obs[ccfg["pert_col"]].astype(str)
    rng = np.random.default_rng(ccfg["seed"])
    perts = sorted(set(tg.unique()) - {ccfg["control"]})
    # 资格校验(ML专家B): 每靶可用细胞数必须 ≥ min_cells_per_pert, 否则 DE gate 会静默清空
    min_cells = int(ccfg.get("min_cells_per_pert", 50))
    tg_np = tg.to_numpy()
    counts = pd.Series(tg_np).value_counts()
    eligible = [p for p in perts if counts.get(p, 0) >= min_cells]
    dropped = len(perts) - len(eligible)
    if dropped:
        print(f"[{ctx}] eligibility gate: {len(eligible)}/{len(perts)} perts have >= {min_cells} cells "
              f"({dropped} dropped)", flush=True)
    chosen = list(rng.choice(eligible, size=min(ccfg["n_perts"], len(eligible)), replace=False))
    keep_idx = []
    for p in chosen + [ccfg["control"]]:
        idx = np.where(tg_np == p)[0]
        take = rng.choice(idx, size=min(n_cells if p != ccfg["control"] else n_ntc, len(idx)),
                          replace=False)
        keep_idx.extend(take.tolist())
    keep_idx = sorted(keep_idx)
    try:
        Xsub = sp.csr_matrix(a.X[keep_idx])   # backed 行子集（只读所需行）
        sub_obs = a.obs.iloc[keep_idx][[ccfg["pert_col"]]].rename(
            columns={ccfg["pert_col"]: "target_gene"}).reset_index(drop=True)
        sub_var = pd.DataFrame(index=pd.Index(a.var_names.astype(str)))
        sub = ad.AnnData(X=Xsub, obs=sub_obs, var=sub_var)
    finally:
        a.file.close()
    sub.write_h5ad(ref, compression="gzip")
    (out / "targets.json").write_text(json.dumps(chosen))
    print(f"[{ctx}] reference ready: {sub.shape}, {len(chosen)} perts", flush=True)
    return ref


def ntc_mean(a: ad.AnnData, control: str) -> np.ndarray:
    mask = a.obs["target_gene"].astype(str).to_numpy() == control
    X = sp.csr_matrix(a[mask].X)
    return np.rint(np.asarray(X.mean(axis=0)).ravel())


def run_celleval(pred: Path, ref: Path, ccfg: dict, outdir: Path) -> pd.DataFrame:
    cmd = [str(ENV_BIN / "cell-eval2"), "run", "-ap", str(pred), "-ar", str(ref),
           "--preset", "vcc2026", "--pert-col", "target_gene", "--control", ccfg["control"],
           "--input-type", "counts", "--set", "de.backend=pdex", "-o", str(outdir)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return pd.read_csv(outdir / "agg_results.csv")


def agg_mean(agg: pd.DataFrame) -> dict:
    row = agg[agg["statistic"] == "mean"]
    vals = row.iloc[0].to_dict() if len(row) else {}
    return {m: float(vals.get(m, np.nan)) for m in METRICS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-config", default=str(ROOT / "configs/local_bench/contexts.yaml"))
    ap.add_argument("--model-config", default=str(ROOT / "configs/l1_signature/ws-v1.1-sampled.yaml"))
    ap.add_argument("--contexts", default="P_H1,Q_HepG2,R_Jurkat")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--anchor", action="store_true")
    args = ap.parse_args()

    bench = load_yaml(Path(args.bench_config))
    mcfg = load_yaml(Path(args.model_config))
    sys_path = str(ROOT)
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from src.models.l1_signature.make_predictions import load_delta

    tag = args.tag or f"lam{mcfg.get('lambda_real', 0)}" + \
        ("_sampled" if mcfg.get("cell_sampling") else "_meanrows")
    out = OUT_DIR / tag
    out.mkdir(parents=True, exist_ok=True)

    per_ctx = {}
    for ctx in args.contexts.split(","):
        ccfg = bench["contexts"][ctx]
        ref = prepare_reference(ctx, ccfg, bench["n_cells_per_pert"], bench["n_ntc_ref"])
        a = ad.read_h5ad(ref)
        control = ccfg["control"]
        mean = ntc_mean(a, control)
        panel = pd.Index(a.var_names.astype(str))
        targets = [t for t in a.obs["target_gene"].astype(str).unique() if t != control]

        delta, rep = load_delta(mcfg, panel, targets)
        dmap = delta.set_index(["target_gene", "gene"])["delta"]
        hits = len(set(targets) & set(dmap.index.get_level_values(0)))

        # 预测（与 L1 相同逻辑: 采样或均值行 ⊕ λ·Δ）
        rng = np.random.default_rng(mcfg.get("seed", 0))
        sampled = bool(mcfg.get("cell_sampling", False))
        if sampled:
            mask = a.obs["target_gene"].astype(str).to_numpy() == control
            Xntc = sp.csr_matrix(a[mask].X)
        blocks = []
        pred_perts = [control] + targets  # cell-eval2 要求集合一致（含对照）
        for p in pred_perts:
            if sampled:
                M = Xntc[rng.choice(Xntc.shape[0], bench["n_cells_per_pert"], replace=True)].copy()
            else:
                M = sp.csr_matrix(np.tile(mean, (bench["n_cells_per_pert"], 1)))
            if p in dmap.index.get_level_values(0):
                d = dmap.loc[p].reindex(panel).to_numpy()
                f = np.power(2.0, mcfg["lambda_real"] * np.nan_to_num(d, nan=0.0)).astype(np.float32)
                M = M.multiply(f.reshape(1, -1)).tocsr()
                M.data = np.clip(np.rint(M.data), 0, None).astype(np.float32)
                M.eliminate_zeros()
            blocks.append(M)
        X = sp.vstack(blocks, format="csr").astype(np.float32)
        X.eliminate_zeros()
        pred = ad.AnnData(X=X, obs=pd.DataFrame(
            {"target_gene": np.repeat(pred_perts, bench["n_cells_per_pert"]), "context": ctx},
            index=[f"p{i}_c{j}" for i in range(len(pred_perts))
                   for j in range(bench["n_cells_per_pert"])]))
        pred.var_names = panel.to_numpy()
        pred.var.index.name = None
        pp = out / f"{ctx}_pred.h5ad"
        pred.write_h5ad(pp, compression="gzip")

        agg = run_celleval(pp, ref, ccfg, out / f"{ctx}_eval")
        per_ctx[ctx] = agg_mean(agg)
        per_ctx[ctx]["coverage"] = hits
        print(f"[{ctx}] " + " ".join(f"{m.split('.')[-1]}={v:.4f}" for m, v in per_ctx[ctx].items()
                                     if m != "coverage"), flush=True)

    # 官方聚合口径: 六指标无权平均 → 上下文无权平均（nmae 未做锚点缩放，仅相对比较用）
    overall = {}
    for m in METRICS:
        vals = [per_ctx[c][m] for c in per_ctx if not np.isnan(per_ctx[c][m])]
        overall[m] = float(np.mean(vals)) if vals else np.nan
    summary = {"tag": tag, "model_config": args.model_config, "per_context": per_ctx,
               "overall_raw": overall}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"overall_raw": overall}, indent=2))


if __name__ == "__main__":
    main()
