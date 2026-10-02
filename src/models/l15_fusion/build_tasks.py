"""L1.5 step1: 构建统一词表、各上下文任务表、NTC basal、签名 store。

用法: python src/models/l15_fusion/build_tasks.py [--contexts gwps,k562_ess,rpe1,jurkat,hepg2]
      python src/models/l15_fusion/build_tasks.py --aux-tahoe   # v3.5: aux 任务表 + basal reindex
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.l15_fusion.common import (build_sig_store, build_task_table, build_vocab,
                                          load_basal, load_cfg, work_dir)


def build_aux(cfg: dict, vocab: pd.Index, cache: Path) -> None:
    """v3.5 aux（tahoe 药扰）: 每上下文任务表 + basal（plate12 var 序 → vocab 序, log1p-CPM）。

    aux 上下文无 CRISPRi 签名文件——任务表直接读 cfg["aux"]["sig_parquet"]，
    绝不经 sig_file()（其 gwps 回退会污染）。
    """
    aux = cfg["aux"]
    rng = np.random.default_rng(cfg["sampling"]["seed"])
    topk, randk = cfg["sampling"]["top_k"], cfg["sampling"]["rand_k"]
    df_all = pd.read_parquet(aux["sig_parquet"], columns=["target_gene", "gene", "logfc"])
    df_all = df_all[df_all["gene"].isin(set(vocab))]  # 词表外基因无特征/嵌入语义
    meta = {"n_raw_rows": int(len(pd.read_parquet(aux["sig_parquet"], columns=["target_gene"]))),
            "vocab_hits": int(len(df_all)), "contexts": {}}
    for ctx in aux["contexts"]:
        prefix = aux["map"][ctx] + "|"  # cell_line → tahoe 组键前缀 (CVCL_xxxx|drug)
        df = df_all[df_all["target_gene"].str.startswith(prefix)].copy()
        if len(df) == 0:
            raise ValueError(f"aux[{ctx}]: 前缀 {prefix} 在 {aux['sig_parquet']} 无行")
        df = df.sample(frac=1.0, random_state=int(rng.integers(2 ** 31)))  # 打散平局
        df["_abs"] = df["logfc"].abs()
        df = df.sort_values(["target_gene", "_abs"], ascending=[True, False], kind="mergesort")
        grp = df.groupby("target_gene", sort=False)
        rank = grp.cumcount().to_numpy()
        size = grp["target_gene"].transform("size").to_numpy()
        is_top = rank < topk
        rest_ok = (rank >= topk) & (size > topk)
        u = rng.random(len(df))
        take_prob = np.where(size > topk, randk / np.maximum(size - topk, 1), 0.0)
        keep = is_top | (rest_ok & (u < take_prob))
        out = df.loc[keep, ["target_gene", "gene", "logfc"]].reset_index(drop=True)
        out.to_parquet(cache / f"tasks_{ctx}.parquet")

        # ---- basal: raw npy (plate12 var 序) → log1p-CPM → vocab 序（缺失基因 0）
        raw_p = Path(aux["basal_raw"][ctx])
        if raw_p.resolve() == (cache / f"basal_{ctx}.npy").resolve():
            raise ValueError(f"aux[{ctx}]: basal_raw 与缓存同路径会覆盖 raw（禁止）")
        raw = np.load(raw_p).astype(np.float64)
        import anndata as ad
        a = ad.read_h5ad(aux["basal_raw_order"], backed="r")
        raw_genes = pd.Index(a.var_names.astype(str))
        a.file.close()
        if len(raw) != len(raw_genes):
            raise ValueError(f"aux[{ctx}]: basal raw len {len(raw)} != plate12 vars {len(raw_genes)}")
        s = raw.sum()
        lg = np.log1p(raw / max(s, 1e-9) * 1e6).astype(np.float32)  # CPM 规约消量纲
        pos = vocab.get_indexer(raw_genes)
        outv = np.zeros(len(vocab), np.float32)
        ok = pos >= 0
        outv[pos[ok]] = lg[ok]
        np.save(cache / f"basal_{ctx}.npy", outv)
        meta["contexts"][ctx] = {
            "prefix": prefix, "groups": int(out.target_gene.nunique()),
            "rows": int(len(out)), "vocab_hits_rows": int(len(df)),
            "basal_nonzero_vocab": int(ok.sum()),
            "basal_log1pcpm_median": float(np.median(outv[outv > 0]))}
        print(f"[aux {ctx}] groups={meta['contexts'][ctx]['groups']} "
              f"task_rows={len(out)} basal_vocab_hits={int(ok.sum())}/{len(vocab)}", flush=True)
    import json
    (cache / "aux_build_meta.json").write_text(json.dumps(meta, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="gwps,k562_ess,rpe1,jurkat,hepg2")
    ap.add_argument("--aux-tahoe", action="store_true",
                    help="只跑 aux（tahoe 药扰）任务表 + basal reindex，跳过主上下文")
    args = ap.parse_args()
    cfg = load_cfg()
    cache = work_dir(cfg)
    t0 = time.time()
    vocab = build_vocab(cfg)
    print(f"[{time.time()-t0:.0f}s] vocab={len(vocab)}", flush=True)
    if args.aux_tahoe:
        if not cfg.get("aux"):
            raise ValueError("--aux-tahoe 需要 cfg['aux'] 配置（v35.yaml）")
        build_aux(cfg, vocab, cache)
        print("build_tasks aux-tahoe complete", flush=True)
        return
    for c in args.contexts.split(","):
        t = time.time()
        tb = build_task_table(c, cfg, vocab, cache)
        print(f"[{time.time()-t0:.0f}s] tasks[{c}]: {len(tb)} rows, "
              f"{tb.target_gene.nunique()} targets ({time.time()-t:.0f}s)", flush=True)
        t = time.time()
        load_basal(c, vocab, cache)
        print(f"[{time.time()-t0:.0f}s] basal[{c}] done ({time.time()-t:.0f}s)", flush=True)
    for c in args.contexts.split(","):
        t = time.time()
        s = build_sig_store(c, vocab, cache)
        print(f"[{time.time()-t0:.0f}s] sigstore[{c}]: {len(s['keys'])} entries "
              f"({time.time()-t:.0f}s)", flush=True)
    print("build_tasks complete", flush=True)


if __name__ == "__main__":
    main()
