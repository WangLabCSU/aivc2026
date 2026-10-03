"""fusion v3.6 = v3.5 Δ̂ × L1.8 置信度 v2 权重 w_j（逐基因乘，保持 (target,gene) 长表结构）。

对两组文件做加权（原文件不动，产 *_confv2 副本）：
  1) LOCO-HepG2 bench Δ: data/interim/l15_fusion/v35/pred_s1bench_hepg2_v35_fused.parquet
     → pred_s1bench_hepg2_v35_confv2.parquet（供 run_bench.py --pred-tag 消费，Q_HepG2 gate）
  2) A/B/C 提交 Δ: data/interim/l15_fusion/v36/delta_submit_{A,B,C}.parquet
     （供 build_submission_v1.py --steps build 消费，λ2.5 载体）

用法: python src/models/l1_signature/apply_conf_v2_fusion.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
W_PATH = ROOT / "data/interim/confidence_v2.parquet"
V35 = ROOT / "data/interim/l15_fusion/v35"
V36 = ROOT / "data/interim/l15_fusion/v36"
NEUTRAL = 0.7


def load_w() -> pd.Series:
    w = pd.read_parquet(W_PATH)
    return pd.Series(w["w"].to_numpy(np.float32), index=w["gene"].astype(str))


def apply_w(df: pd.DataFrame, w: pd.Series) -> tuple[pd.DataFrame, dict]:
    wv = df["gene"].astype(str).map(w).fillna(NEUTRAL).to_numpy(np.float32)
    out = df.copy()
    out["delta"] = (out["delta"].to_numpy(np.float32) * wv).astype(np.float32)
    rep = {"rows": int(len(out)), "mean_w": round(float(wv.mean()), 4),
           "frac_neutral_0.7": round(float((wv == NEUTRAL).mean()), 4),
           "absmean_before": round(float(df["delta"].abs().mean()), 4),
           "absmean_after": round(float(out["delta"].abs().mean()), 4),
           "sig_rows_before": int((df["delta"].abs() > 0.25).sum()),
           "sig_rows_after": int((out["delta"].abs() > 0.25).sum())}
    return out, rep


def main() -> None:
    w = load_w()
    report = {"weight_table": str(W_PATH), "table_genes": int(len(w))}
    # 1) LOCO-HepG2 bench Δ
    src = V35 / "pred_s1bench_hepg2_v35_fused.parquet"
    dst = V35 / "pred_s1bench_hepg2_v35_confv2.parquet"
    if not dst.exists():
        df = pd.read_parquet(src)
        out, rep = apply_w(df, w)
        out.to_parquet(dst, index=False)
    else:
        rep = {"cached": True}
    report["loco_hepg2"] = {"src": str(src), "dst": str(dst), **rep}
    print("loco_hepg2:", report["loco_hepg2"], flush=True)

    # 2) A/B/C 提交 Δ（v36 workdir，不动 v35 原件）
    V36.mkdir(parents=True, exist_ok=True)
    for cx in ["A", "B", "C"]:
        src = V35 / f"delta_submit_{cx}.parquet"
        dst = V36 / f"delta_submit_{cx}.parquet"
        if dst.exists():
            print(f"[{cx}] cached {dst}", flush=True)
            continue
        df = pd.read_parquet(src)
        out, rep = apply_w(df, w)
        out.to_parquet(dst, index=False)
        report[f"submit_{cx}"] = rep
        print(f"[{cx}] rows={rep['rows']} mean_w={rep['mean_w']} "
              f"absmean {rep['absmean_before']}→{rep['absmean_after']} "
              f"sig {rep['sig_rows_before']}→{rep['sig_rows_after']}", flush=True)
    # vocab 复用（build_submission_v1 的 build_vocab 需要）
    vocab_src, vocab_dst = V35 / "vocab.json", V36 / "vocab.json"
    if not vocab_dst.exists():
        vocab_dst.write_text(vocab_src.read_text())
    (V36 / "_apply_conf_report.json").write_text(
        __import__("json").dumps(report, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
