"""双家族 Δ̂ 级 ensemble α 扫描（Q_HepG2, λ4.5）：dualsrc-deep 启发式 × fusion v3.5。

Δ̂_ens = α·Δ̂_heur + (1-α)·Δ̂_fusion（union 语义：加权 skipna 均值，单源独有对保留单源值）。
- Δ̂_heur: dualsrc 语义逐字复刻（fid_dirprior base 变体）——gwps×hct116_all 等权均值、
  单源保留、clip±4、未覆盖靶 gwps generic 全额兜底。
- Δ̂_fusion: 复用 data/interim/l15_fusion/v35/pred_s2bench_hepg2_v35_fused.parquet；
  并用 feat_bench_hepg2.npz + xgb_s2_35 + mlp_s2_35_noemb(α=0.5) 复算校验。
- pred 构建: run_bench.build_pred 逐字逻辑（NTC bootstrap 400/pert seed0 ⊕ 2^(λ·Δ̂)，
  逐元素 max_fold×原始计数帽，rint；rng 逐 pert 抽取顺序与 run_bench 一致）。
  λ4.5 / max_fold 8（与 fusion λ 曲线 l15v35s2_fused_lam* 同参，家族内配对可比）。

用法:
  python scripts/ens_alpha_scan.py deltas            # 5α delta parquet + pred h5ad
  python scripts/ens_alpha_scan.py eval --alpha 0.25 # 单 α cell-eval2 run
  python scripts/ens_alpha_scan.py summary           # 汇总 5α + fusion λ 曲线 + affine 预测
"""
from __future__ import annotations

import argparse
import json
import os
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
OUT = ROOT / "outputs/metrics/local_bench"
BENCH = ROOT / "data/interim/local_bench/Q_HepG2"
V35 = ROOT / "data/interim/l15_fusion/v35"
SCAN = V35 / "ensemble_scan"
SIG = ROOT / "data/interim/signatures"
REF = BENCH / "ref.h5ad"
LAM = 4.5
MAX_FOLD = 8.0
N_CELLS = 400
SEED = 0
CLIP = 4.0
CTRL = "non-targeting"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CFG = ["--preset", "vcc2026", "--pert-col", "target_gene", "--control", CTRL,
       "--input-type", "counts", "--set", "de.backend=pdex"]


def tag_of(a: float) -> str:
    lam = f"{LAM:g}".replace(".", "")
    return f"ensA{round(a * 100):03d}_lam{lam}"


def load_fusion_delta() -> tuple[pd.DataFrame, dict]:
    """复用缓存的 fused parquet，并用特征缓存+双模型复算校验。"""
    cached = pd.read_parquet(V35 / "pred_s2bench_hepg2_v35_fused.parquet")
    rep: dict = {"cached_rows": len(cached),
                 "cached_targets": int(cached.target_gene.nunique())}
    try:
        os.environ["L15_CFG"] = "configs/l15_fusion/v35.yaml"
        sys.path.insert(0, str(ROOT))
        from src.models.l15_fusion.common import load_cfg, work_dir
        from src.models.l15_fusion.predict_loco import predict_mlp, predict_xgb
        cfg = load_cfg()
        cache = work_dir(cfg)
        z = np.load(V35 / "feat_bench_hepg2.npz", allow_pickle=False)
        X, TG, GN = z["X"], z["target"].astype(str), z["gene"].astype(str)
        px = predict_xgb(cfg, cache, X, 2, model_tag="s2_35")
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pm = predict_mlp(cfg, cache, X, 2,
                         np.zeros(len(X), np.int64), np.zeros(len(X), np.int64), dev,
                         pt_tag="s2_35_noemb")
        pf = np.clip(np.nan_to_num(0.5 * px + 0.5 * pm, nan=0.0), -CLIP, CLIP)
        d2 = pd.DataFrame({"target_gene": TG, "gene": GN, "delta": pf.astype(np.float32)})
        key = ["target_gene", "gene"]
        m = cached.merge(d2, on=key, suffixes=("_c", "_r"), how="inner")
        diff = float((m["delta_c"] - m["delta_r"]).abs().max())
        rep.update({"recheck_rows": len(m), "recheck_maxabsdiff": diff,
                    "recheck_pass": bool(diff < 1e-4)})
        print(f"[fusion recheck] max|Δ|={diff:.2e} rows={len(m)}", flush=True)
    except Exception as e:  # 复算失败不阻塞主流程
        rep.update({"recheck_error": repr(e)})
        print(f"[fusion recheck] FAILED: {e!r}（用缓存）", flush=True)
    return cached, rep


def heur_delta(targets: list[str], panel: pd.Index) -> pd.DataFrame:
    """dualsrc 语义：等权均值、单源保留、clip±4、generic 兜底（fid base 变体逐字）。"""
    def src_table(name: str) -> pd.DataFrame:
        df = pd.read_parquet(SIG / f"{name}.parquet",
                             filters=[("target_gene", "in", targets)])
        df = df[df["gene"].isin(panel)]
        g = df.groupby(["target_gene", "gene"], sort=False)["logfc"].mean().reset_index()
        print(f"[heur:{name}] rows={len(g)} targets={g.target_gene.nunique()}", flush=True)
        return g
    g = src_table("gwps")
    h = src_table("hct116_all")
    m = g.merge(h, on=["target_gene", "gene"], how="outer", suffixes=("_g", "_h"))
    a, b = m["logfc_g"].to_numpy(), m["logfc_h"].to_numpy()
    both = np.isfinite(a) & np.isfinite(b)
    v = np.where(both, (np.nan_to_num(a) + np.nan_to_num(b)) / 2.0,
                 np.where(np.isfinite(a), a, b))
    d = m.loc[:, ["target_gene", "gene"]].copy()
    d["delta"] = v.astype(np.float32).clip(-CLIP, CLIP)
    covered = set(d.target_gene.unique())
    unc = [t for t in targets if t not in covered]
    if unc:
        gw = pd.read_parquet(SIG / "gwps.parquet", columns=["gene", "logfc"])
        gen = gw.groupby("gene", sort=False)["logfc"].mean()
        gen = gen[gen.index.isin(panel)]
        d = pd.concat([d, pd.DataFrame(
            {"target_gene": np.repeat(unc, len(gen)),
             "gene": np.tile(gen.index.to_numpy(), len(unc)),
             "delta": np.tile(gen.to_numpy(), len(unc)).astype(np.float32)})],
            ignore_index=True)
    print(f"[heur] covered={len(covered)}/{len(targets)} unc={len(unc)} rows={len(d)}",
          flush=True)
    return d


def blend(dh: pd.DataFrame, df: pd.DataFrame, alpha: float, panel: pd.Index) -> pd.DataFrame:
    """加权 skipna 均值（pivot 上做，单源独有对保留单源值）。"""
    ph = dh.pivot(index="target_gene", columns="gene", values="delta") \
        .reindex(columns=panel)
    pf = df.pivot(index="target_gene", columns="gene", values="delta") \
        .reindex(columns=panel)
    idx = ph.index.union(pf.index)
    ph = ph.reindex(index=idx)
    pf = pf.reindex(index=idx)
    phv, pfv = ph.to_numpy(), pf.to_numpy()
    wh, wf = alpha, 1.0 - alpha
    num = wh * np.nan_to_num(phv, nan=0.0) + wf * np.nan_to_num(pfv, nan=0.0)
    den = wh * np.isfinite(phv) + wf * np.isfinite(pfv)
    ens = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    out = pd.DataFrame({"target_gene": np.repeat(ph.index.to_numpy(), len(panel)),
                        "gene": np.tile(panel.to_numpy(), len(ph)),
                        "delta": ens.astype(np.float32).ravel()})
    return out[np.isfinite(out["delta"])].reset_index(drop=True)


def build_pred(delta: pd.DataFrame, outdir: Path) -> Path:
    """run_bench.build_pred 逐字（rng 抽取顺序一致）。"""
    a = ad.read_h5ad(REF)
    panel = pd.Index(a.var_names.astype(str))
    targets = json.loads((BENCH / "targets.json").read_text())
    mask = a.obs["target_gene"].astype(str).to_numpy() == CTRL
    Xntc = sp.csr_matrix(a[mask].X)
    del a
    rng = np.random.default_rng(SEED)
    dmap = delta.set_index(["target_gene", "gene"])["delta"]
    pred_perts = [CTRL] + targets
    blocks = []
    for p in pred_perts:
        M = Xntc[rng.choice(Xntc.shape[0], N_CELLS, replace=True)].copy()
        if p in dmap.index.get_level_values(0):
            dd = dmap.loc[p]
            if isinstance(dd.index, pd.MultiIndex):
                dd = dd.xs(p, level=0)
            dv = dd.reindex(panel).to_numpy()
            f = np.power(2.0, LAM * np.nan_to_num(dv, nan=0.0)).astype(np.float32)
            orig = M.data.copy()
            M.data = np.minimum(M.data * f[M.indices], MAX_FOLD * orig)
        M.data = np.clip(np.rint(M.data), 0, None).astype(np.float32)
        M.eliminate_zeros()
        blocks.append(M)
    X = sp.vstack(blocks, format="csr").astype(np.float32)
    X.eliminate_zeros()
    pred = ad.AnnData(X=X, obs=pd.DataFrame(
        {"target_gene": np.repeat(pred_perts, N_CELLS), "context": "hepg2"},
        index=[f"p{i}_c{j}" for i in range(len(pred_perts)) for j in range(N_CELLS)]))
    pred.var_names = panel.to_numpy()
    pred.var.index.name = None
    outdir.mkdir(parents=True, exist_ok=True)
    pp = outdir / "Q_HepG2_pred.h5ad"
    pred.write_h5ad(pp, compression="gzip")
    return pp


def cmd_deltas() -> None:
    SCAN.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _r = ad.read_h5ad(REF, backed="r")
    panel = pd.Index(_r.var_names.astype(str))
    targets = json.loads((BENCH / "targets.json").read_text())
    _r.file.close()
    dfus, frep = load_fusion_delta()
    (SCAN / "fusion_recheck.json").write_text(json.dumps(frep, indent=2))
    dh = heur_delta(targets, panel)
    dh.to_parquet(SCAN / "delta_heur.parquet")
    dfus.to_parquet(SCAN / "delta_fusion.parquet")
    # 家族 Δ̂ 谱对比（Q 100 靶 × panel 对齐后）
    ph = dh[dh.target_gene.isin(targets)].pivot(
        index="target_gene", columns="gene", values="delta") \
        .reindex(columns=panel).to_numpy().ravel()
    pf = dfus.pivot(index="target_gene", columns="gene", values="delta") \
        .reindex(columns=panel).to_numpy().ravel()
    both = np.isfinite(ph) & np.isfinite(pf)
    corr = float(np.corrcoef(ph[both], pf[both])[0, 1])
    print(f"[families] corr(heur,fusion)={corr:.4f} npairs={int(both.sum())} "
          f"absmean h/f={np.nanmean(np.abs(ph[both])):.3f}/{np.nanmean(np.abs(pf[both])):.3f}",
          flush=True)
    for al in ALPHAS:
        de = blend(dh, dfus, al, panel)
        dp = SCAN / f"delta_a{al}.parquet"
        de.to_parquet(dp)
        pp = build_pred(de, OUT / tag_of(al))
        print(f"[a={al}] delta rows={len(de)} absmean={de.delta.abs().mean():.4f} "
              f"nnz_pred={pp} ({time.time()-t0:.0f}s)", flush=True)


def cmd_eval(alpha: float) -> None:
    tag = tag_of(alpha)
    outdir = OUT / tag
    pp = outdir / "Q_HepG2_pred.h5ad"
    ev = outdir / "Q_HepG2_eval"
    cmd = [str(ENV_BIN / "cell-eval2"), "run", "-ap", str(pp), "-ar", str(REF),
           *CFG, "-o", str(ev)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"[{tag}] eval done", flush=True)


def cmd_summary() -> None:
    tags = [tag_of(a) for a in ALPHAS]
    subprocess.run([str(ENV_BIN / "python"), str(ROOT / "src/eval/bench_score.py"),
                    "--tags", ",".join(tags), "--contexts", "Q_HepG2",
                    "--out", str(SCAN / "anchor_normalized_ens_scan.json")], check=True)
    print("[summary] anchor_normalized_ens_scan.json written", flush=True)


def main() -> None:
    global LAM
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["deltas", "eval", "point", "summary"])
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=LAM)
    args = ap.parse_args()
    if args.mode == "deltas":
        cmd_deltas()
    elif args.mode == "eval":
        cmd_eval(args.alpha)
    elif args.mode == "point":
        LAM = args.lam
        de = pd.read_parquet(SCAN / f"delta_a{args.alpha}.parquet")
        pp = build_pred(de, OUT / tag_of(args.alpha))
        print(f"[point] pred built: {pp}", flush=True)
        cmd_eval(args.alpha)
    else:
        cmd_summary()


if __name__ == "__main__":
    main()
