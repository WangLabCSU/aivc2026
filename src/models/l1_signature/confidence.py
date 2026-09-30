"""L1.7 方向置信度加权：跨数据集响应基因级方向一致率 → 每基因权重 w_g。

免疫学动机：K562 gwps 签名迁移到竞赛 A/B/C 时，缺失模块（p53 轴/IFN/谱系）方向迁移失效。
一个基因的扰动响应方向若在多个独立数据集/细胞系中一致（p53 靶基因处处下调等），
其 Δ 方向更可信；方向分裂（同基因在不同上下文反向响应）则不可信。

投票规则（IMM 审定）：
- 仅当基因在某数据集某扰动下显著响应（|logFC| >= sig_threshold，log1p-CP10K 空间）才计票；
- 同数据集内多扰动同基因可能方向分裂 → 按数据集聚合：数据集级方向 = 该基因全部
  显著扰动 logfc 的均值符号（每个数据集一票 ±1）；
- N = 有票数据集数；agree = 与多数方向一致的票数；
- w = 2*agree/N - 1，clip 到 [0,1]（全票一致=1，对半分裂=0）；
- N < 3 时中性 + 软收缩：w' = beta + (1-beta)*w（beta=0.5，弱证据基因回到半置信）；
- 权重表里没有的基因 → w' = beta。

用法（独立诊断）:
  python src/models/l1_signature/confidence.py [--threshold 0.25] [--beta 0.5] \
      [--exclude hepg2] [--panel-genes] [--out <csv>]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SIG_DIR = ROOT / "data/interim/signatures"

# 默认投票数据集（gwps 是 Δ 的源签名本身，不参与投票：置信度应衡量独立于源的一致性）
DEFAULT_VOTE_DATASETS = ["norman", "k562_ess", "rpe1", "hepg2", "jurkat"]
MIN_VOTES_FOR_CONFIDENT = 3  # N < 3 触发 beta 软收缩


def _sig_path(name: str) -> Path:
    """优先全量签名 <name>_all.parquet，缺失时回退 300 靶版 <name>.parquet。"""
    full = SIG_DIR / f"{name}_all.parquet"
    return full if full.exists() else SIG_DIR / f"{name}.parquet"


def compute_direction_votes(parquet_paths: list[str | Path], gene_panel=None,
                            sig_threshold: float = 0.25, beta: float = 0.5,
                            exclude_datasets: list[str] | None = None,
                            invert: bool = False) -> tuple[pd.DataFrame, dict]:
    """逐数据集计票，返回 (每基因投票明细表, 报告)。

    明细表列: [gene, n_votes(N), agree, w_raw, w]，仅含有票基因。
    invert=True 为诊断模式：N>=3 的基因 w=1-w_raw（不一致基因升权、一致基因降权），
    N<3 基因保持中性 1.0（与无置信基线等权），用于验证一致性-迁移收益的相关符号。
    """
    exclude_datasets = set(exclude_datasets or [])
    votes: dict[str, list[int]] = {}
    used, skipped = [], []
    for p in parquet_paths:
        name = Path(p).stem.removesuffix("_all")
        if name in exclude_datasets:
            skipped.append(name)
            continue
        df = pd.read_parquet(p, columns=["target_gene", "gene", "logfc"])
        if gene_panel is not None:
            df = df[df["gene"].isin(gene_panel)]
        sig = df[df["logfc"].abs() >= sig_threshold]
        used.append({"dataset": name, "path": str(p), "rows": len(df),
                     "sig_rows": int(len(sig)),
                     "sig_genes": int(sig["gene"].nunique()) if len(sig) else 0})
        if sig.empty:
            continue
        # 数据集级方向 = 显著扰动 logfc 的均值符号（一数据集一票）
        direction = np.sign(sig.groupby("gene", sort=False)["logfc"].mean())
        for g, s in direction.items():
            if s != 0:
                votes.setdefault(g, []).append(int(s))

    rows = []
    for g, v in votes.items():
        n = len(v)
        agree = max(v.count(1), v.count(-1))
        w_raw = float(np.clip(2.0 * agree / n - 1.0, 0.0, 1.0))
        if invert:
            w = float(1.0 - w_raw) if n >= MIN_VOTES_FOR_CONFIDENT else 1.0
        else:
            w = w_raw if n >= MIN_VOTES_FOR_CONFIDENT else float(beta + (1.0 - beta) * w_raw)
        rows.append({"gene": g, "n_votes": n, "agree": agree, "w_raw": w_raw, "w": w})
    detail = (pd.DataFrame(rows).sort_values(["n_votes", "w"], ascending=[False, False])
              if rows else
              pd.DataFrame(columns=["gene", "n_votes", "agree", "w_raw", "w"]))
    report = {"used": used, "excluded": sorted(exclude_datasets),
              "n_genes_with_votes": len(detail),
              "n_genes_Nge3": int((detail["n_votes"] >= MIN_VOTES_FOR_CONFIDENT).sum())
              if len(detail) else 0}
    return detail, report


def compute_direction_confidence(parquet_paths: list[str | Path], gene_panel=None,
                                 sig_threshold: float = 0.25, beta: float = 0.5,
                                 exclude_datasets: list[str] | None = None,
                                 invert: bool = False) -> pd.Series:
    """主入口：返回 pd.Series（index=gene，值=w'∈[0,1]）。无票基因不出现（上游填 beta）。"""
    detail, _ = compute_direction_votes(parquet_paths, gene_panel, sig_threshold, beta,
                                        exclude_datasets, invert)
    if detail.empty:
        return pd.Series(dtype=np.float64, name="w")
    return detail.set_index("gene")["w"]


def vote_datasets(names: list[str] | None = None) -> list[Path]:
    """把数据集名解析为签名 parquet 路径（默认 DEFAULT_VOTE_DATASETS）。"""
    return [_sig_path(n) for n in (names or DEFAULT_VOTE_DATASETS)]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--datasets", default=",".join(DEFAULT_VOTE_DATASETS))
    ap.add_argument("--exclude", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    paths = vote_datasets(args.datasets.split(","))
    detail, report = compute_direction_votes(paths, None, args.threshold, args.beta,
                                             [x for x in args.exclude.split(",") if x])
    import json

    print(json.dumps(report, indent=2, ensure_ascii=False))
    hist = detail.groupby(pd.cut(detail["w"], bins=[-0.01, 0.0, 0.25, 0.5, 0.75, 0.99, 1.0],
                                 include_lowest=True)).size()
    print("w 分布（含 N<3 软收缩后）:\n", hist.to_string())
    print("N 分布:\n", detail["n_votes"].value_counts().sort_index().to_string())
    if args.out:
        detail.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
