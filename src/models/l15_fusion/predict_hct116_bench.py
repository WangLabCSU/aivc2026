"""fusion (v2 或 v1h 消融) 对 S_HCT116 全深度 bench 的 Δ̂ 预测 + 采样载体 + cell-eval2。

流程:
  1) ref = data/interim/local_bench/S_HCT116/ref.h5ad（先由 src/eval/local_bench.py 重建）；
     targets.json 100 靶 × ref 面板(38,606) → FeatureBuilder(row_ctx='hct116').transform
     （v2: sig_hct116 置 NaN 防泄漏；basal_j = v2 basal_hct116）
  2) Δ̂ = 0.5·XGB(s2) + 0.5·MLP(s2_noemb)，clip ±4；面板外(词表外)基因 Δ=0
  3) 采样载体: ref NTC 细胞 bootstrap 400 ⊕ 2^(λ·Δ̂)，逐元素 max_fold×原始计数帽，
     rint→csr；pred 含 NTC 块（label 取 contexts.yaml 的 control），轴=ref 同序
  4) cell-eval2 run（--control 取 contexts.yaml）

用法:
  python src/models/l15_fusion/predict_hct116_bench.py                     # v2, λ2.5
  python src/models/l15_fusion/predict_hct116_bench.py --cfg configs/l15_fusion/v1h.yaml \
      --pred-tag s2bench_hct116_v1 --bench-tag l15v1_hct116_lam2.5         # v1 消融
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
ENV_BIN = Path("/data/home/wsx/miniconda3/envs/aivc2026/bin")
BENCH = ROOT / "data/interim/local_bench"
OUT = ROOT / "outputs/metrics/local_bench"
CTX = "S_HCT116"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/l15_fusion/v2.yaml")
    ap.add_argument("--lam", type=float, default=2.5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--pred-tag", default="s2bench_hct116_v2")
    ap.add_argument("--bench-tag", default="l15v2_hct116_lam2.5")
    ap.add_argument("--n-cells", type=int, default=400)
    ap.add_argument("--max-fold", type=float, default=8.0)
    ap.add_argument("--clip", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    import yaml
    ccfg = yaml.safe_load(open(ROOT / "configs/local_bench/contexts.yaml"))["contexts"][CTX]
    control = str(ccfg["control"])
    t0 = time.time()

    sys.path.insert(0, str(ROOT))
    import os
    os.environ["L15_CFG"] = args.cfg
    from src.models.l15_fusion.common import FeatureBuilder, build_vocab, load_cfg, work_dir
    from src.models.l15_fusion.predict_loco import predict_mlp, predict_xgb

    cfg = load_cfg()
    cache = work_dir(cfg)
    vocab = build_vocab(cfg)
    print(f"cfg={args.cfg} vocab={len(vocab)} workdir={cache} lam={args.lam}", flush=True)

    ref_path = BENCH / CTX / "ref.h5ad"
    targets = json.loads((BENCH / CTX / "targets.json").read_text())
    import anndata as ad
    _r = ad.read_h5ad(ref_path, backed="r")
    panel = pd.Index(_r.var_names.astype(str))
    _r.file.close()
    TG = np.repeat(np.array(targets, dtype="U"), len(panel))
    GN = np.tile(panel.to_numpy().astype("U"), len(targets))

    # ---- Δ̂
    dp = cache / f"pred_{args.pred_tag}.parquet"
    if dp.exists():
        d = pd.read_parquet(dp)
        print(f"delta loaded from cache: {dp} rows={len(d)}", flush=True)
    else:
        fb = FeatureBuilder(cfg, vocab, cache)
        Xs = []
        for s in range(0, len(TG), 2_000_000):
            Xs.append(fb.transform("hct116", TG[s:s + 2_000_000], GN[s:s + 2_000_000]))
            print(f"  features {Xs[-1].shape} ({time.time()-t0:.0f}s)", flush=True)
        X = np.concatenate(Xs)
        del Xs
        px = predict_xgb(cfg, cache, X, 2)
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pm = predict_mlp(cfg, cache, X, 2, vocab.get_indexer(pd.Index(GN)).astype(np.int64),
                         vocab.get_indexer(pd.Index(TG)).astype(np.int64), dev,
                         pt_tag="s2_noemb")
        del X
        pf = (args.alpha * px + (1 - args.alpha) * pm).astype(np.float32)
        d = pd.DataFrame({"target_gene": TG, "gene": GN,
                          "delta": np.clip(np.nan_to_num(pf, nan=0.0), -args.clip, args.clip)})
        d.to_parquet(dp)
        print(f"delta saved rows={len(d)} absmean={d.delta.abs().mean():.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 采样载体 pred（NTC bootstrap ⊕ 2^(λ·Δ̂)，max_fold 帽）
    a = ad.read_h5ad(ref_path)
    mask = a.obs["target_gene"].astype(str).to_numpy() == control
    Xntc = sp.csr_matrix(a[mask].X)
    del a
    rng = np.random.default_rng(args.seed)
    dmap = d.set_index(["target_gene", "gene"])["delta"]
    pred_perts = [control] + list(targets)
    blocks = []
    for p in pred_perts:
        M = Xntc[rng.choice(Xntc.shape[0], args.n_cells, replace=True)].copy()
        if p in dmap.index.get_level_values(0):
            dv = dmap.loc[p].reindex(panel).to_numpy()
            f = np.power(2.0, args.lam * np.nan_to_num(dv, nan=0.0)).astype(np.float32)
            orig = M.data.copy()
            M.data = np.minimum(M.data * f[M.indices], args.max_fold * orig)
        M.data = np.clip(np.rint(M.data), 0, None).astype(np.float32)
        M.eliminate_zeros()
        blocks.append(M)
    Xp = sp.vstack(blocks, format="csr").astype(np.float32)
    Xp.eliminate_zeros()
    outdir = OUT / args.bench_tag
    outdir.mkdir(parents=True, exist_ok=True)
    pred = ad.AnnData(X=Xp, obs=pd.DataFrame(
        {"target_gene": np.repeat(pred_perts, args.n_cells), "context": CTX},
        index=[f"p{i}_c{j}" for i in range(len(pred_perts)) for j in range(args.n_cells)]))
    pred.var_names = panel.to_numpy()
    pred.var.index.name = None
    pp = outdir / f"{CTX}_pred.h5ad"
    pred.write_h5ad(pp, compression="gzip")
    print(f"pred built: {pp} ({time.time()-t0:.0f}s)", flush=True)
    if args.skip_eval:
        return
    cmd = [str(ENV_BIN / "cell-eval2"), "run", "-ap", str(pp), "-ar", str(ref_path),
           "--preset", "vcc2026", "--pert-col", "target_gene", "--control", control,
           "--input-type", "counts", "--set", "de.backend=pdex",
           "-o", str(outdir / f"{CTX}_eval")]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"[done] eval → {outdir / f'{CTX}_eval'} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
