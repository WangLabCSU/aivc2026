"""L1.5 step2: 提取基础模型基因嵌入并 PCA→128。

- SE-600M: protein_embeddings.pt = {symbol: (5120,)}，直接按 symbol 取
- Tahoe-x1-70M: model.gene_encoder.embedding.weight (62720,512)，vocab=ENSG→id；
  symbol→ENSG 用 mygene.info querymany（失败则跳过 Tahoe）

输出: <workdir>/emb_se.npy (V,128), emb_tahoe.npy (V,128) + *_genes 对齐 vocab。
用法: python src/models/l15_fusion/extract_embeddings.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.l15_fusion.common import ROOT, build_vocab, load_cfg, work_dir


def pca_reduce(X: np.ndarray, dim: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    ok = np.isfinite(X).all(axis=1)
    p = PCA(n_components=dim, random_state=0)
    Z = np.zeros((len(X), dim), np.float32)
    Z[ok] = p.fit_transform(X[ok]).astype(np.float32)
    print(f"  PCA {X.shape[1]}→{dim}, explained={p.explained_variance_ratio_.sum():.3f}",
          flush=True)
    return Z


def se600m(vocab: np.ndarray, dim: int, cache: Path) -> None:
    import torch
    out = cache / "emb_se.npy"
    if out.exists():
        print("emb_se exists", flush=True)
        return
    t = time.time()
    d = torch.load(str(ROOT / "data/public/models/SE-600M/protein_embeddings.pt"),
                   weights_only=False, map_location="cpu")
    d = {k: v.numpy().astype(np.float32) for k, v in d.items()}
    print(f"  SE loaded {len(d)} genes ({time.time()-t:.0f}s)", flush=True)
    idx = {g: i for i, g in enumerate(d.keys())}
    ndim = next(iter(d.values())).shape[0]
    rows, have = [], []
    missing = 0
    for i, g in enumerate(vocab.tolist()):
        j = idx.get(g)
        if j is None:
            rows.append(np.zeros(ndim, np.float32))
            missing += 1
        else:
            rows.append(d[g])
    X = np.stack(rows)
    print(f"  SE coverage {len(vocab)-missing}/{len(vocab)}", flush=True)
    Z = pca_reduce(X, dim)
    np.save(out, Z)
    (cache / "emb_se_meta.json").write_text(json.dumps(
        {"source": "SE-600M protein_embeddings", "dim_in": int(X.shape[1]),
         "missing": missing, "pca": dim}))
    print(f"emb_se done ({time.time()-t:.0f}s)", flush=True)


def tahoe(vocab: np.ndarray, dim: int, cache: Path, cfg: dict) -> None:
    out = cache / "emb_tahoe.npy"
    if out.exists():
        print("emb_tahoe exists", flush=True)
        return
    if not cfg["embeddings"].get("use_tahoe", True):
        return
    import torch
    t0 = time.time()
    tdir = Path(cfg["embeddings"]["tahoe_dir"])
    v = json.load(open(tdir / "vocab.json"))
    syms = vocab.tolist()
    # symbol → ENSG：本地来源 = 各上下文 h5ad var.ensembl_id（已核实 11920/11920 全覆盖）
    import anndata as ad
    import pandas as pd
    cfg2 = load_cfg()
    sym2ensg: dict[str, str] = {}
    for f in set(list(cfg2["basal_h5ad"].values())):
        a = ad.read_h5ad(ROOT / f, backed="r")
        vd = a.var[["ensembl_id"]].astype(str)
        vd = vd[vd.index.isin(set(syms)) & vd["ensembl_id"].str.startswith("ENSG")]
        m = dict(zip(vd.index.tolist(), vd["ensembl_id"].tolist()))
        for k, val in m.items():
            sym2ensg.setdefault(k, val)
        a.file.close()
    print(f"  local ENSG map {len(sym2ensg)}/{len(syms)}", flush=True)
    from safetensors.torch import load_file
    st = load_file(str(tdir / "model.safetensors"))
    E = st["model.gene_encoder.embedding.weight"].numpy().astype(np.float32)
    id_of = v  # vocab.json 本身就是 token(ENSG) -> id
    X = np.zeros((len(vocab), E.shape[1]), np.float32)
    missing = 0
    for i, g in enumerate(syms):
        e = sym2ensg.get(g)
        tid = id_of.get(e, -1) if e else -1
        if tid < 0:
            missing += 1
        else:
            X[i] = E[tid]
    print(f"  tahoe coverage {len(vocab)-missing}/{len(vocab)}", flush=True)
    Z = pca_reduce(X, dim)
    np.save(out, Z)
    (cache / "emb_tahoe_meta.json").write_text(json.dumps(
        {"source": "Tahoe-x1-70M gene_encoder embedding", "dim_in": int(E.shape[1]),
         "missing": missing, "pca": dim}))
    print(f"emb_tahoe done ({time.time()-t0:.0f}s)", flush=True)


def main() -> None:
    cfg = load_cfg()
    cache = work_dir(cfg)
    vocab = np.asarray(build_vocab(cfg).tolist(), dtype="U")
    dim = int(cfg["embeddings"]["pca_dim"])
    se600m(vocab, dim, cache)
    tahoe(vocab, dim, cache, cfg)


if __name__ == "__main__":
    main()
