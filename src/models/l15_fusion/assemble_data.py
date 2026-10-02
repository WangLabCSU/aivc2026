"""L1.5 step3: 组装并缓存特征矩阵。

scopes:
  train           — 4 训练上下文任务表全量 (ctx 列区分, stage1 取 ctx!=jurkat)
  proxy_jurkat    — jurkat 全 2393 靶 × (top200|true|+300 随机基因)，proxy 指标用
  proxy_hepg2     — hepg2 全靶 proxy 表
  bench_jurkat    — R_Jurkat bench 100 靶 × jurkat 面板（复用 local_bench ref/targets）
  bench_hepg2     — Q_HepG2 bench 100 靶 × hepg2 面板

用法: python src/models/l15_fusion/assemble_data.py --which train,proxy_jurkat,bench_hepg2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.l15_fusion.common import (FeatureBuilder, build_task_table, build_vocab,
                                          load_cfg, sig_file, work_dir)

BENCH = Path("/data/resource/wanglab/aivc/aivc2026/data/interim/local_bench")


def _chunked_transform(fb, row_ctx, targets, genes, CH=2_000_000):
    Xs = []
    for s in range(0, len(targets), CH):
        Xs.append(fb.transform(row_ctx, targets[s:s + CH], genes[s:s + CH]))
    return np.concatenate(Xs)


def build_train(cfg, fb, vocab, cache) -> None:
    p = cache / "feat_train.npz"
    if p.exists():
        print("feat_train exists", flush=True)
        return
    Xs, ys, ctxs, gps, tps, ws = [], [], [], [], [], []
    names = None
    aux = cfg.get("aux") or {}
    aux_w = float(aux.get("weight", 0.3))
    aux_ctxs = list(aux.get("contexts", []))
    for c in cfg["train_contexts"]:
        t = time.time()
        tb = build_task_table(c, cfg, vocab, cache)
        tg = tb["target_gene"].to_numpy()
        gn = tb["gene"].to_numpy()
        X = _chunked_transform(fb, c, tg, gn)
        Xs.append(X)
        ys.append(tb["logfc"].to_numpy(np.float32))
        ctxs.append(np.full(len(tb), cfg["train_contexts"].index(c), np.int8))
        gps.append(vocab.get_indexer(pd.Index(gn)))
        tps.append(vocab.get_indexer(pd.Index(tg)))
        ws.append(np.ones(len(tb), np.float32))
        print(f"  train[{c}]: {X.shape} ({time.time()-t:.0f}s)", flush=True)
    for i, c in enumerate(aux_ctxs):  # v3.5: tahoe 药扰辅助行（权重 aux_w, has_sig=0）
        t = time.time()
        tb = pd.read_parquet(cache / f"tasks_{c}.parquet")
        tg = tb["target_gene"].to_numpy()
        gn = tb["gene"].to_numpy()
        X = _chunked_transform(fb, c, tg, gn)
        Xs.append(X)
        ys.append(tb["logfc"].to_numpy(np.float32))
        ctxs.append(np.full(len(tb), len(cfg["train_contexts"]) + i, np.int8))
        gps.append(vocab.get_indexer(pd.Index(gn)))
        tps.append(np.zeros(len(tb), np.int64))  # 药扰组不是基因 → E 查表占位 0
        ws.append(np.full(len(tb), aux_w, np.float32))
        print(f"  train[aux:{c}]: {X.shape} w={aux_w} ({time.time()-t:.0f}s)", flush=True)
    X = np.concatenate(Xs)
    gene_pos_all = np.concatenate(gps)
    np.savez(p, X=X, y=np.concatenate(ys), ctx=np.concatenate(ctxs),
             gpos=gene_pos_all.astype(np.int32), names=np.array(fb.names()),
             w=np.concatenate(ws).astype(np.float32),
             tpos=np.concatenate(tps).astype(np.int32))
    np.save(cache / "tpos_train.npy", np.concatenate(tps).astype(np.int32))
    del Xs
    n_aux = sum(int((np.concatenate(ctxs) == len(cfg["train_contexts"]) + i).sum())
                for i in range(len(aux_ctxs)))
    print(f"feat_train saved: {X.shape} aux_rows={n_aux} ({time.time()-t:.0f}s total)", flush=True)


def build_proxy(cfg, fb, vocab, cache, ctx: str) -> None:
    p = cache / f"feat_proxy_{ctx}.npz"
    if p.exists():
        print(f"feat_proxy_{ctx} exists", flush=True)
        return
    t = time.time()
    sig = pd.read_parquet(sig_file(ctx))
    rng = np.random.default_rng(0)
    top_n, rand_n = 200, 300
    parts = []
    for tg, sub in sig.groupby("target_gene", sort=False):
        sub = sub[sub["gene"].isin(vocab)]
        if len(sub) == 0:
            continue
        sub = sub.reindex(sub["logfc"].abs().sort_values(ascending=False).index)
        top = sub.head(top_n)
        rest = sub.iloc[top_n:]
        take = rest.sample(n=min(rand_n, len(rest)), random_state=0)
        parts.append(pd.concat([top, take])[["target_gene", "gene", "logfc"]])
    tb = pd.concat(parts, ignore_index=True)
    tg = tb["target_gene"].to_numpy()
    gn = tb["gene"].to_numpy()
    X = _chunked_transform(fb, ctx, tg, gn)
    np.savez(p, X=X, y=tb["logfc"].to_numpy(np.float32), names=np.array(fb.names()),
             target=tg.astype("U"), gene=gn.astype("U"))
    print(f"feat_proxy_{ctx}: {X.shape} ({time.time()-t:.0f}s)", flush=True)


def build_bench(cfg, fb, vocab, cache, ctx: str) -> None:
    bname = {"jurkat": "R_Jurkat", "hepg2": "Q_HepG2"}[ctx]
    p = cache / f"feat_bench_{ctx}.npz"
    if p.exists():
        print(f"feat_bench_{ctx} exists", flush=True)
        return
    t = time.time()
    targets = json.loads((BENCH / bname / "targets.json").read_text())
    import anndata as ad
    a = ad.read_h5ad(BENCH / bname / "ref.h5ad", backed="r")
    panel = np.asarray(a.var_names.astype(str).tolist())
    a.file.close()
    TG = np.repeat(np.array(targets, dtype="U"), len(panel))
    GN = np.tile(panel, len(targets))
    X = _chunked_transform(fb, ctx, TG, GN)
    np.savez(p, X=X, names=np.array(fb.names()), target=TG.astype("U"),
             gene=GN.astype("U"), targets=np.array(targets, dtype="U"), panel=panel.astype("U"))
    print(f"feat_bench_{ctx}: {X.shape} ({time.time()-t:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True)
    args = ap.parse_args()
    cfg = load_cfg()
    cache = work_dir(cfg)
    vocab = build_vocab(cfg)
    fb = FeatureBuilder(cfg, vocab, cache)
    t0 = time.time()
    for w in args.which.split(","):
        if w == "train":
            build_train(cfg, fb, vocab, cache)
        elif w.startswith("proxy_"):
            build_proxy(cfg, fb, vocab, cache, w.split("_", 1)[1])
        elif w.startswith("bench_"):
            build_bench(cfg, fb, vocab, cache, w.split("_", 1)[1])
        else:
            raise ValueError(w)
    print(f"assemble done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
