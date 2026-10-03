"""fid 方向先验子集选择（v0.2 fid 专项）: gwps×hct116_all 双源方向一致性 → Δ̂ 二值子集筛选。

设计（区别于已失败的幅度缩水 L1.7/L1.8——不缩量程, 只做子集选择与有效 λ 变更）:
共享靶上每 (target,gene) 响应分三类:
  双源同向  a·b>0        → 保留 (a+b)/2（w=1, 与 dualsrc 基线一致）
  双源反向  a·b<0        → 整段剔除（w=0, 预测 factor=1）
  单源显著  |v|>=0.25    → 保留但 λ 减半（delta×0.5; λ3.5→1.75）
  单源不显著            → 剔除
未覆盖靶 gwps generic 兜底同"单源"规则处理。
变体: v1=全三分类; v2=消融（仅反向剔除, 单源全额保留）——分离"方向剔除"与"λ减半"两效应。
应用: dualsrc-deep λ3.5 Δ̂ → Q_HepG2 bench vs 双源基线（0.3182, 同路重验基线）。

用法: python scripts/fid_dirprior_v1.py [--lam 3.5] [--sig 0.25] [--variants v1,v2]
"""
from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/metrics/local_bench"
WORK = ROOT / "data/interim/fid_dirprior"
REF = ROOT / "data/interim/local_bench/Q_HepG2/ref.h5ad"
SIG = ROOT / "data/interim/signatures"
LAM = 3.5
MAX_FOLD = 12.0          # dualsrc.yaml 基线值
CTRL = "non-targeting"
CFG = ["--preset", "vcc2026", "--pert-col", "target_gene", "--input-type", "counts",
       "--set", "de.backend=pdex", "--control", CTRL]


def src_table(name: str, targets: list[str], panel: pd.Index) -> pd.DataFrame:
    f = SIG / f"{name}.parquet"
    df = pd.read_parquet(f, filters=[("target_gene", "in", targets)])
    df = df[df["gene"].isin(panel)]
    g = df.groupby(["target_gene", "gene"], sort=False)["logfc"].mean().reset_index()
    print(f"[{name}] rows={len(g)} targets={g['target_gene'].nunique()}", flush=True)
    return g


def generic_table(panel: pd.Index) -> pd.Series:
    gw = pd.read_parquet(SIG / "gwps.parquet", columns=["gene", "logfc"])
    gen = gw.groupby("gene", sort=False)["logfc"].mean()
    return gen[gen.index.isin(panel)]


def build_delta(m: pd.DataFrame, targets: list[str], panel: pd.Index, gen: pd.Series,
                sig: float, variant: str) -> tuple[pd.DataFrame, dict]:
    """m: outer merge [target_gene, gene, a(gwps), b(hct116)]。"""
    a, b = m["a"].to_numpy(), m["b"].to_numpy()
    both = np.isfinite(a) & np.isfinite(b)
    prod = np.where(both, np.sign(a) * np.sign(b), np.nan)
    delta = np.full(len(m), np.nan)
    n = {"both_same": 0, "both_opposite": 0, "single_sig": 0, "single_weak": 0}
    if variant == "v1":      # 全三分类: 反向剔除; 单源显著 λ/2, 弱剔除
        keep_same = both & (prod > 0)
        delta[keep_same] = ((a[keep_same] + b[keep_same]) / 2.0)
        # 单源（含一方恰为 0）: 用非 NaN 方
        single = ~both
        v = np.where(np.isfinite(a), a, b)
        vs = single & (np.abs(np.nan_to_num(v)) >= sig)
        delta[vs] = v[vs] * 0.5
        n["both_same"] = int(keep_same.sum())
        n["both_opposite"] = int((both & (prod < 0)).sum())
        n["single_sig"] = int(vs.sum())
        n["single_weak"] = int(single.sum() - vs.sum())
    else:                    # v2 消融: 仅反向剔除, 单源全额保留（基线语义）
        keep_same = both & (prod > 0)
        delta[keep_same] = (a[keep_same] + b[keep_same]) / 2.0
        single = ~both
        v = np.where(np.isfinite(a), a, b)
        delta[single] = v[single]
        n["both_same"] = int(keep_same.sum())
        n["both_opposite"] = int((both & (prod < 0)).sum())
        n["single_sig"] = int(single.sum())
        n["single_weak"] = 0
    out = m.loc[np.isfinite(delta), ["target_gene", "gene"]].copy()
    out["delta"] = delta[np.isfinite(delta)].astype(np.float32)
    out["delta"] = out["delta"].clip(-4.0, 4.0)
    # 未覆盖靶: gwps generic 兜底, 同单源规则
    covered = set(out["target_gene"].unique())
    unc = [t for t in targets if t not in covered]
    if unc and variant == "v1":
        gv = gen.to_numpy()
        sel = np.abs(gv) >= sig
        gen2 = pd.DataFrame({"target_gene": np.repeat(unc, int(sel.sum())),
                             "gene": np.tile(gen.index.to_numpy()[sel], len(unc)),
                             "delta": (gv[sel] * 0.5).astype(np.float32)})
    elif unc:
        gen2 = pd.DataFrame({"target_gene": np.repeat(unc, len(gen)),
                             "gene": np.tile(gen.index.to_numpy(), len(unc)),
                             "delta": gen.to_numpy().astype(np.float32)})
    else:
        gen2 = None
    stats = {"variant": variant, "sig": sig, "covered_targets": len(covered),
             "uncovered_targets": len(unc), "rows_real": len(out),
             "classes": n,
             "generic_rows": 0 if gen2 is None else len(gen2),
             "absmean": float(out["delta"].abs().mean())}
    if gen2 is not None:
        out = pd.concat([out, gen2], ignore_index=True)
    return out, stats


def bench_delta(delta: pd.DataFrame, tag: str, lam: float) -> dict:
    a = ad.read_h5ad(REF)
    panel = pd.Index(a.var_names.astype(str))
    targets = [t for t in a.obs["target_gene"].astype(str).unique() if t != CTRL]
    dmap = delta.set_index(["target_gene", "gene"])["delta"]
    mask = a.obs["target_gene"].astype(str).to_numpy() == CTRL
    Xntc = sp.csr_matrix(a[mask].X)
    del a
    rng = np.random.default_rng(0)
    pred_perts = [CTRL] + targets
    blocks = []
    for p in pred_perts:
        M = Xntc[rng.choice(Xntc.shape[0], 400, replace=True)].copy()
        if p in dmap.index.get_level_values(0):
            dd = dmap.loc[p]
            if isinstance(dd.index, pd.MultiIndex):
                dd = dd.xs(p, level=0)
            dv = dd.reindex(panel).to_numpy()
            f = np.power(2.0, lam * np.nan_to_num(dv, nan=0.0)).astype(np.float32)
            orig = M.data.copy()
            M.data = np.minimum(M.data * f[M.indices], MAX_FOLD * orig)
        M.data = np.clip(np.rint(M.data), 0, None).astype(np.float32)
        M.eliminate_zeros()
        blocks.append(M)
    X = sp.vstack(blocks, format="csr").astype(np.float32)
    X.eliminate_zeros()
    pred = ad.AnnData(X=X, obs=pd.DataFrame(
        {"target_gene": np.repeat(pred_perts, 400), "context": "Q_HepG2"},
        index=[f"p{i}_c{j}" for i in range(len(pred_perts)) for j in range(400)]))
    pred.var_names = panel.to_numpy()
    pred.var.index.name = None
    outdir = OUT / tag
    outdir.mkdir(parents=True, exist_ok=True)
    pp = outdir / "Q_HepG2_pred.h5ad"
    pred.write_h5ad(pp, compression="gzip")
    ev = outdir / "Q_HepG2_eval"
    subprocess.run([str(ENV_BIN / "cell-eval2"), "run", "-ap", str(pp), "-ar", str(REF),
                    *CFG, "-o", str(ev)], check=True)
    from src.eval.bench_score import score_tag_ctx
    bundle = OUT / "bundles" / "Q_HepG2"
    res = score_tag_ctx(tag, "Q_HepG2", bundle if (bundle / "manifest.json").exists() else None)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--sig", type=float, default=0.25)
    ap.add_argument("--variants", default="base,v1,v2")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    a0 = ad.read_h5ad(REF, backed="r")
    panel = pd.Index(a0.var_names.astype(str))
    targets = [t for t in a0.obs["target_gene"].astype(str).unique() if t != CTRL]
    a0.file.close()
    print(f"panel={len(panel)} targets={len(targets)}", flush=True)
    g = src_table("gwps", targets, panel)
    h = src_table("hct116_all", targets, panel)
    m = g.merge(h, on=["target_gene", "gene"], how="outer", suffixes=("_g", "_h"))
    m = m.rename(columns={"logfc_g": "a", "logfc_h": "b"})
    gen = generic_table(panel)
    print(f"merge rows={len(m)} ({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for var in args.variants.split(","):
        tag = f"fid_dp_{var}"
        sp_ = WORK / f"delta_{var}.parquet"
        if var == "base":
            # 基线重验: dualsrc 语义（等权均值, 单源保留, generic 全额）— 不经先验
            both = np.isfinite(m["a"]) & np.isfinite(m["b"])
            d = m.loc[:, ["target_gene", "gene"]].copy()
            v = np.where(both, (m["a"].fillna(0) + m["b"].fillna(0)) / 2.0,
                         m["a"].fillna(m["b"]))
            d["delta"] = v.astype(np.float32).clip(-4.0, 4.0)
            gv = gen.to_numpy()
            covered = set(d["target_gene"].unique())
            unc = [t for t in targets if t not in covered]
            d = pd.concat([d, pd.DataFrame(
                {"target_gene": np.repeat(unc, len(gen)),
                 "gene": np.tile(gen.index.to_numpy(), len(unc)),
                 "delta": np.tile(gv, len(unc)).astype(np.float32)})], ignore_index=True)
            st = {"variant": "base", "covered": len(covered), "rows": len(d),
                  "absmean": float(d["delta"].abs().mean())}
        else:
            d, st = build_delta(m, targets, panel, gen, args.sig, var)
            d.to_parquet(sp_)
        (WORK / f"stats_{var}.json").write_text(json.dumps(st, indent=2))
        print(f"[{tag}] stats={st}", flush=True)
        res = bench_delta(d, tag, args.lam)
        results[var] = {"stats": st, "overall": res["overall"],
                        "per_metric": res["per_metric"]}
        print(f"[{tag}] overall={res['overall']:.4f} "
              f"fid={res['per_metric'].get('fid')} pds={res['per_metric'].get('pds')}", flush=True)
    results["_baseline_official_ledger"] = {
        "tag": "dualsrc_deep", "avg_score": 0.3182,
        "per_metric": {"pds": 0.8181, "mse": 0.0, "nmae": 0.0439, "fid": 0.7753,
                       "reach": 0.1961, "jac": 0.0761}}
    (WORK / "dirprior_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: (v["overall"] if isinstance(v, dict) and "overall" in v else v)
                      for k, v in results.items()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
