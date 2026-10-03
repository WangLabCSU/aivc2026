"""L1.8 fid 方向置信度加权 v2（保弱去强改进版）：Δ 双源自身 (gwps, hct116_all) 方向一致率。

与 L1.7（confidence.py，负结果）的区别：
- L1.7 用 5 个外部数据集投票、w∈[0,1] 双峰、mean_w≈0.76 的幅度缩水被 bench 证伪
  （reach 塌掉盖过 jac 小赚；反转诊断证明一致基因是信号本体）。
- v2 只用构建 Δ̂ 的两个源自身（gwps=K562 CRISPRi deep, hct116_all=xatlas HCT116 CRISPRi deep）
  做跨源方向一致率：两源同向 → 该基因的 dual-source 均值方向可信（w=1，不动它——吸取保强教训）；
  两源反向 → 均值是互相抵消的混合，方向不可信（w=0.5）；证据不足 → 中性 0.7。
- w 下限 0.3（非 0），保弱不灭signal；fid 官方最大失分项 -0.199 的目标就是去掉"方向分裂基因"
  对 fid yield 的拖累，同时几乎不动强一致基因的幅度。

投票规则（per 响应基因 j，阈值 |logfc|>=0.25）：
- 源 s 显著响应 := 该源内存在 (target,j) 行 |logfc|>=thr；源级方向 = 显著行 logfc 均值的符号
  （与 L1.7 逐数据集聚合口径一致）；
- 两源均显著：同向 → 两票 agree=+1（agree_sum=2）；反向 → 一 agree 一 disagree
  （agree_sum=+1-1=0）。N_votes=2；
- 仅单源显著 → 0 票（"仅单源显著→0"），N_votes=1；
- w = clip(0.5 + 0.5*agree_sum/N_votes, 0.3, 1.0)；N_votes<2 → w=0.7（中性）。

用法:
  python src/models/l1_signature/confidence_v2.py [--threshold 0.25] \
      [--sources gwps,hct116_all] [--out data/interim/confidence_v2.parquet]
产物: parquet [gene, n_sig_gwps, n_sig_hct116, dir_gwps, dir_hct116, n_votes, agree_sum, w]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SIG_DIR = ROOT / "data/interim/signatures"
DEFAULT_SOURCES = ["gwps", "hct116_all"]
NEUTRAL_W = 0.7
W_MIN, W_MAX = 0.3, 1.0


def source_gene_direction(path: Path, threshold: float = 0.25) -> pd.DataFrame:
    """单源显著行聚合 → [gene, n_sig, dir]（dir=显著行 logfc 均值符号）。

    大 parquet（hct116_all 6.7e8 行）用 pyarrow 谓词下推只读显著行，避免全量进内存。
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    tab = pq.read_table(path, columns=["gene", "logfc"],
                        filters=[[("logfc", ">=", threshold)],
                                 [("logfc", "<=", -threshold)]])
    df = tab.to_pandas()
    if df.empty:
        return pd.DataFrame(columns=["gene", "n_sig", "dir"])
    g = df.groupby("gene", sort=False)["logfc"].agg(n_sig="size", mean="mean")
    g = g[g["mean"] != 0.0]
    return pd.DataFrame({"gene": g.index.to_numpy(),
                         "n_sig": g["n_sig"].to_numpy(np.int64),
                         "dir": np.sign(g["mean"].to_numpy()).astype(np.int8)})


def compute_confidence_v2(sources: list[str] = None, threshold: float = 0.25) -> tuple[pd.DataFrame, dict]:
    """双源方向一致率 → w 表。返回 (detail[gene,n_sig_*,dir_*,n_votes,agree_sum,w], report)。"""
    sources = sources or DEFAULT_SOURCES
    dirs = {}
    report = {"sources": {}, "threshold": threshold}
    for name in sources:
        p = SIG_DIR / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        d = source_gene_direction(p, threshold)
        dirs[name] = d.set_index("gene")
        report["sources"][name] = {"path": str(p), "sig_genes": int(len(d)),
                                   "sig_rows": int(d["n_sig"].sum())}
        print(f"[{name}] sig genes={len(d)} rows={int(d['n_sig'].sum())}", flush=True)

    a, b = sources[0], sources[1]
    m = dirs[a].join(dirs[b], how="outer", lsuffix=f"_{a}", rsuffix=f"_{b}")
    for c in ["n_sig", "dir"]:
        for s in sources:
            col = f"{c}_{s}"
            if col not in m.columns:
                m[col] = pd.Series(np.nan, index=m.index)
    m["n_sig"] = m[[f"n_sig_{s}" for s in sources]].fillna(0).sum(axis=1).astype(np.int64)

    n_sig_a = m[f"n_sig_{a}"].fillna(0).to_numpy()
    n_sig_b = m[f"n_sig_{b}"].fillna(0).to_numpy()
    da = m[f"dir_{a}"].fillna(0).to_numpy()
    db = m[f"dir_{b}"].fillna(0).to_numpy()

    both = (n_sig_a > 0) & (n_sig_b > 0)
    agree_sum = np.zeros(len(m))
    agree_sum[both & (da == db)] = 2.0          # 两票 agree
    agree_sum[both & (da != db)] = 0.0          # 一 agree 一 disagree
    n_votes = np.where(both, 2, (n_sig_a > 0) | (n_sig_b > 0)).astype(int)  # 单源显著 N=1
    w = np.where(n_votes >= 2,
                 np.clip(0.5 + 0.5 * agree_sum / np.maximum(n_votes, 1), W_MIN, W_MAX),
                 NEUTRAL_W)

    detail = pd.DataFrame({
        "gene": m.index.to_numpy(),
        f"n_sig_{a}": n_sig_a.astype(np.int64),
        f"n_sig_{b}": n_sig_b.astype(np.int64),
        f"dir_{a}": da.astype(np.int8),
        f"dir_{b}": db.astype(np.int8),
        "n_votes": n_votes,
        "agree_sum": agree_sum,
        "w": np.round(w, 4),
    }).reset_index(drop=True)
    report["n_genes"] = int(len(detail))
    report["counts"] = {
        "both_agree(w=1.0)": int(((detail["n_votes"] == 2) & (detail["agree_sum"] == 2)).sum()),
        "both_disagree(w=0.5)": int(((detail["n_votes"] == 2) & (detail["agree_sum"] == 0)).sum()),
        "single_source(w=0.7)": int((detail["n_votes"] == 1).sum()),
        "w_mean": round(float(detail["w"].mean()), 4),
    }
    return detail, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    ap.add_argument("--out", default=str(ROOT / "data/interim/confidence_v2.parquet"))
    args = ap.parse_args()

    detail, report = compute_confidence_v2(args.sources.split(","), args.threshold)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    detail.to_parquet(out, index=False)
    print(f"wrote {out} ({len(detail)} genes); w hist:\n", detail["w"].value_counts().to_string())


if __name__ == "__main__":
    main()
