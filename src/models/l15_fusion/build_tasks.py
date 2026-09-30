"""L1.5 step1: 构建统一词表、各上下文任务表、NTC basal、签名 store。

用法: python src/models/l15_fusion/build_tasks.py [--contexts gwps,k562_ess,rpe1,jurkat,hepg2]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.l15_fusion.common import (build_sig_store, build_task_table, build_vocab,
                                          load_basal, load_cfg, work_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="gwps,k562_ess,rpe1,jurkat,hepg2")
    args = ap.parse_args()
    cfg = load_cfg()
    cache = work_dir(cfg)
    t0 = time.time()
    vocab = build_vocab(cfg)
    print(f"[{time.time()-t0:.0f}s] vocab={len(vocab)}", flush=True)
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
