"""ctxmatch: 上下文匹配双源签名 —— A/B/C 各取最像的 Replogle 上下文签名替代 pooled gwps。

动机: pooled gwps 把跨上下文响应平均化, 稀释逐扰动特异性 (排行榜主战场 pds/reach)。
匹配 (basal 相关): A→jurkat, B→rpe1, C→hct116 深签名单独用 (它本就是深源)。
语义: 双源等权 skipna 均值 (与 dualsrc 一致), 仅源1 从 gwps→ctx*_all;
     generic 兜底也换成 ctx* 的 gene-mean。C 的 ctx* = hct116 时退化为 hct116 单源。
注: 本地 bench 上下文 = Replogle 上下文本身, ctxmatch 本地必 oracle 泄漏 → 只做机制校验,
    诚实评估靠官方探针。

用法:
  python scripts/ctxmatch_pred.py --map A=jurkat,B=rpe1,C=hct116 --lam 6.5 \
      --out-stem 20261012_ctxmatch_lam65
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
SIG = ROOT / "data/interim/signatures"
CTRLS = ROOT / "data/raw/controls"
LAM = 6.5
MAX_FOLD = 16.0
N_CELLS = 400
SEED = 0
CLIP = 4.0
CTRL = "non-targeting"


def src_table(name: str, targets: list[str], genes: set[str]) -> pd.DataFrame:
    df = pd.read_parquet(SIG / f"{name}.parquet")
    df = df[df["target_gene"].isin(targets) & df["gene"].isin(genes)]
    g = df.groupby(["target_gene", "gene"], sort=False)["logfc"].mean().reset_index()
    print(f"[{name}] rows={len(g)} targets={g.target_gene.nunique()}", flush=True)
    return g


def build_for_context(cx: str, match: str, panel: pd.Index,
                      targets: list[str]) -> pd.DataFrame:
    """dualsrc 语义, 源1 = match 上下文签名; match=hct116 时单深源。"""
    gs = set(panel)
    a = src_table(match, targets, gs)
    if match == "hct116":
        d = a.rename(columns={"logfc": "delta"})
    else:
        b = src_table("hct116_all", targets, gs)
        m = a.merge(b, on=["target_gene", "gene"], how="outer",
                    suffixes=("_a", "_b"))
        va = pd.to_numeric(m["logfc_a"], errors="coerce").to_numpy(dtype="float64")
        vb = pd.to_numeric(m["logfc_b"], errors="coerce").to_numpy(dtype="float64")
        both = np.isfinite(va) & np.isfinite(vb)
        v = np.where(both, (np.nan_to_num(va) + np.nan_to_num(vb)) / 2.0,
                     np.where(np.isfinite(va), va, vb))
        d = m.loc[:, ["target_gene", "gene"]].copy()
        d["delta"] = v.astype(np.float32).clip(-CLIP, CLIP)
    d["delta"] = d["delta"].clip(-CLIP, CLIP)
    covered = set(d.target_gene.unique())
    unc = [t for t in targets if t not in covered]
    if unc and match != "hct116":
        gw = pd.read_parquet(SIG / f"{match}.parquet", columns=["gene", "logfc"])
        gen = gw.groupby("gene", sort=False)["logfc"].mean()
        gen = gen[gen.index.isin(gs)]
        d = pd.concat([d, pd.DataFrame(
            {"target_gene": np.repeat(unc, len(gen)),
             "gene": np.tile(gen.index.to_numpy(), len(unc)),
             "delta": np.tile(gen.to_numpy(), len(unc)).astype(np.float32)})],
            ignore_index=True)
    print(f"[{cx}<-{match}] covered={len(covered)}/{len(targets)} unc={len(unc)} "
          f"rows={len(d)} absmean={d.delta.abs().mean():.4f}", flush=True)
    return d


def build_pred(deltas: dict[str, pd.DataFrame], out: Path, lam: float) -> None:
    rng = np.random.default_rng(SEED)
    blocks, obs_rec = [], []
    for cx in ["A", "B", "C"]:
        a = ad.read_h5ad(CTRLS / f"context_{cx}.h5ad")
        panel = pd.Index(a.var_names.astype(str))
        targets = pd.read_csv(CTRLS / "pert_counts.csv").iloc[:, 0].tolist()
        mask = a.obs["target_gene"].astype(str).to_numpy() == CTRL
        Xntc = sp.csr_matrix(a[mask].X)
        del a
        dmap = deltas[cx].set_index(["target_gene", "gene"])["delta"]
        for p in [CTRL] + targets:
            M = Xntc[rng.choice(Xntc.shape[0], N_CELLS, replace=True)].copy()
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
            obs_rec.extend([(cx, p, f"{cx}_{i}") for i in range(N_CELLS)])
        del Xntc
    X = sp.vstack(blocks, format="csr").astype(np.float32)
    X.eliminate_zeros()
    obs = pd.DataFrame(obs_rec, columns=["context", "target_gene", "cell_id"])
    pred = ad.AnnData(X=X, obs=obs.set_index("cell_id"))
    pred.var_names = panel.to_numpy()
    pred.var.index.name = None
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(out, compression="gzip")
    print(f"[pred] {out} shape={X.shape}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="A=jurkat,B=rpe1,C=hct116")
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--out-stem", default="20261012_ctxmatch_lam65")
    args = ap.parse_args()
    t0 = time.time()
    mapping = dict(kv.split("=") for kv in args.map.split(","))
    deltas = {}
    targets = pd.read_csv(CTRLS / "pert_counts.csv").iloc[:, 0].tolist()
    a0 = ad.read_h5ad(CTRLS / "context_A.h5ad", backed="r")
    panel = pd.Index(a0.var_names.astype(str))
    a0.file.close()
    for cx in ["A", "B", "C"]:
        deltas[cx] = build_for_context(cx, mapping[cx], panel, targets)
    out = ROOT / "outputs/submissions" / f"{args.out_stem}.h5ad"
    build_pred(deltas, out, args.lam)
    (ROOT / "outputs/metrics" / f"{args.out_stem}.json").write_text(json.dumps(
        {"mapping": mapping, "lam": args.lam, "absmean":
            {c: float(d.delta.abs().mean()) for c, d in deltas.items()}}, indent=1))
    print(f"done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
