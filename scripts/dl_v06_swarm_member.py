#!/usr/bin/env python
"""v0.6 gate4 three-arm swarm member (CPU-only).

三臂消融评测准备 (prereg §5 门槛4):
  arm dl       : DL Δ̂ 单臂 — data/interim/bizard/v06_dl/pred_dl_v06_{ctx}.parquet 原样
  arm birds_dl : per-bird preds + DL → swarm.aggregate(method="soft_rank") 等权融合
  arm birds    : 冻结聚合链 agg_delta_{ctx}.parquet — 只读引用, 永不重算

对 src/models/bizard/swarm.py 只读 (aggregate/soft_rank_average 调用, 零修改);
bench 评分走 bench_bird.py CLI (--delta-dir/--delta-pattern), 同样零修改。
行对齐硬约束: 融合前逐臂 assert 成员 (target_gene, gene) 网格逐位相同。

用法:
  # 1) 融合 (前置: tmp/dl_v06_gen_parquets.py 已产出 v06_dl parquet)
  /data/home/wsx/miniconda3/envs/aivc2026/bin/python scripts/dl_v06_swarm_member.py \
      --arm birds_dl --contexts P_H1,Q_HepG2,R_Jurkat,S_RPE1,S_HCT116
  # 2) 三臂 bench 命令 (默认 dry-run 打印; --run 交由 team-lead 决定执行)
  /data/home/wsx/miniconda3/envs/aivc2026/bin/python scripts/dl_v06_swarm_member.py --arm bench-cmds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
sys.path.insert(0, str(ROOT))
from src.models.bizard.swarm import aggregate, load_cfg, workdir  # noqa: E402

DL_DIR = ROOT / "data/interim/bizard/v06_dl"
FUSED_DIR = DL_DIR / "fused"
BIRD_CFG = ROOT / "configs/bizard/v03.yaml"   # 鸟册 (per-bird pred_{bird}_{ctx}.parquet)
LAM = 2.5                                     # bench_lambda (v03 aggregate.bench_lambda)
BENCH_CTX = ["P_H1", "Q_HepG2", "R_Jurkat", "S_RPE1", "S_HCT116"]


def load_grid(df: pd.DataFrame) -> np.ndarray:
    return df["target_gene"].to_numpy().astype(object) + "\x00" + df["gene"].to_numpy().astype(object)


def fuse_birds_dl(ctx: str, dl_dir: Path, outdir: Path, extra_members: list[Path]) -> Path:
    """per-bird preds + DL → soft_rank_average 等权融合 (row-aligned)."""
    cfg = load_cfg(BIRD_CFG)
    wdir = workdir(cfg)
    members: list[tuple[str, pd.DataFrame]] = []
    for b in cfg["birds"]:
        p = wdir / f"pred_{b['name']}_{ctx}.parquet"
        if p.exists():
            members.append((f"bird:{b['name']}", pd.read_parquet(p, columns=["target_gene", "gene", "delta"])))
        else:
            print(f"[{ctx}] WARN missing bird pred {p.name}", flush=True)
    for p in extra_members:
        members.append((f"extra:{p.parent.name}/{p.name}", pd.read_parquet(p, columns=["target_gene", "gene", "delta"])))
    dl_path = dl_dir / f"pred_dl_v06_{ctx}.parquet"
    members.append(("dl_v06", pd.read_parquet(dl_path)))
    assert len(members) >= 2, f"[{ctx}] need >=2 members, got {len(members)}"

    ref = load_grid(members[0][1])
    for name, df in members[1:]:
        assert len(df) == len(ref) and (load_grid(df) == ref).all(), \
            f"[{ctx}] row-grid mismatch in {name} — soft_rank_average 行对齐约束, 停不产 gate 读数"

    fused = aggregate([df["delta"].to_numpy() for _, df in members], method="soft_rank")
    out = pd.DataFrame({
        "target_gene": members[0][1]["target_gene"].to_numpy(),
        "gene": members[0][1]["gene"].to_numpy(),
        "delta": fused,
    })
    outdir.mkdir(parents=True, exist_ok=True)
    op = outdir / f"bzv06_birds_dl_{ctx}.parquet"
    out.to_parquet(op, index=False)
    print(f"[{ctx}] fused {len(members)} members -> {op.name} rows={len(out)}", flush=True)
    return op


def bench_cmds(tag: str, delta_dir: Path, pattern: str, contexts: str) -> list[str]:
    py = "/data/home/wsx/miniconda3/envs/aivc2026/bin/python"
    # pattern carries a literal {ctx} placeholder (bench_bird formats it per context)
    return [f"{py} src/models/bizard/bench_bird.py --tag {tag} --lam {LAM} "
            f"--contexts {contexts} --delta-dir {delta_dir} "
            f"--delta-pattern '{pattern}'"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["dl", "birds", "birds_dl", "bench-cmds"])
    ap.add_argument("--contexts", default=",".join(BENCH_CTX))
    ap.add_argument("--dl-dir", default=str(DL_DIR))
    ap.add_argument("--extra-members", default="", help="逗号分隔的额外融合成员 parquet 路径")
    args = ap.parse_args()
    contexts = args.contexts.split(",")
    dl_dir = Path(args.dl_dir)

    if args.arm == "birds_dl":
        extra = [Path(p) for p in args.extra_members.split(",") if p]
        for ctx in contexts:
            fuse_birds_dl(ctx, dl_dir, FUSED_DIR, extra)
        return

    if args.arm == "bench-cmds":
        ctxs = args.contexts
        for tag, ddir, pat in [
            ("bzv06dl_arm_dl", dl_dir, "pred_dl_v06_{ctx}.parquet"),
            ("bzv06dl_arm_birds_dl", FUSED_DIR, "bzv06_birds_dl_{ctx}.parquet"),
            ("bzv06dl_arm_birds", ROOT / "data/interim/bizard/v03", "agg_delta_{ctx}.parquet"),
        ]:
            for c in bench_cmds(tag, ddir, pat, ctxs):
                print(c)
        print("\n# arm dl 与 arm birds 也可直接复用冻结 bench 读数 (不重跑): "
              "arm birds = v0.5 谱系既有 bz 系列读数; 重跑仅用于同口径复核。", flush=True)
        return

    raise SystemExit(f"arm {args.arm} 的 parquet 由生成器/冻结链产出, 本脚本不复制 (只读引用)")


if __name__ == "__main__":
    main()
