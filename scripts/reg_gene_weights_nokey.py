#!/usr/bin/env python
"""REV C3 硬前置 — load_delta no-key 5.27M 行逐位回归（gene_weights 键合入验收）。

模板: iter014 sign_fix/dataset_weights 验收（docs/iterations/2026-10-10_iter014_fid三候选证伪.md:
"无新键时与旧聚合逐位一致（max|Δ|=0，5.27M 行回归校验）"）。
基准 config: configs/l1_signature/dualsrc-deep-lam65.yaml（gwps + hct116_all, 无任何可选键,
sign_fix 关闭）。panel = data/raw/controls/gene_names.csv (18,533), targets = pert_counts.csv (300)。

用法:
  python scripts/reg_gene_weights_nokey.py baseline   # 键合入前: 快照现行为输出
  python scripts/reg_gene_weights_nokey.py compare    # 键合入后: 重跑并比对, 要求 max|Δ|=0
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.l1_signature.make_predictions import load_delta  # noqa: E402

OUT = ROOT / "outputs/gene_weights_regression"
CFG = ROOT / "configs/l1_signature/dualsrc-deep-lam65.yaml"


def run_load_delta():
    cfg = yaml.safe_load(CFG.read_text())
    assert not cfg.get("gene_weights") and not cfg.get("sign_fix") \
        and not cfg.get("dataset_weights") and not cfg.get("confidence_file"), \
        "回归基准 config 必须无任何可选键（no-key 口径）"
    panel = pd.Index(pd.read_csv(ROOT / "data/raw/controls/gene_names.csv")["gene_name"].astype(str))
    targets = pd.read_csv(ROOT / "data/raw/controls/pert_counts.csv")["target_gene"].astype(str).tolist()
    return load_delta(cfg, panel, targets)


def main(phase: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    real, report = run_load_delta()
    if phase == "baseline":
        real.to_parquet(OUT / "snapshot_delta.parquet", index=False)
        (OUT / "snapshot_report.json").write_text(
            json.dumps(report, indent=2, default=str, sort_keys=True))
        sha = hashlib.sha256((OUT / "snapshot_delta.parquet").read_bytes()).hexdigest()
        (OUT / "snapshot_sha256.txt").write_text(sha + "\n")
        print(f"[baseline] rows={len(real)} sha256={sha[:16]}...")
        print(f"[baseline] covered={report['covered_targets']} uncovered={report['uncovered_targets']}")
        return 0

    # compare: 快照完整性 → 键集/行数 → 逐位 max|Δ| → report 全等
    sha0 = (OUT / "snapshot_sha256.txt").read_text().strip()
    sha_now = hashlib.sha256((OUT / "snapshot_delta.parquet").read_bytes()).hexdigest()
    snap_ok = sha_now == sha0
    snap = pd.read_parquet(OUT / "snapshot_delta.parquet")
    rep0 = json.loads((OUT / "snapshot_report.json").read_text())
    m = snap.merge(real, on=["target_gene", "gene"], how="outer", suffixes=("_0", "_1"),
                   indicator=True)
    n_both = int((m["_merge"] == "both").sum())
    only0 = int((m["_merge"] == "left_only").sum())
    only1 = int((m["_merge"] == "right_only").sum())
    both = m[m["_merge"] == "both"]
    no_nan = bool(both["delta_0"].notna().all() and both["delta_1"].isna().sum() == 0
                  and real["delta"].notna().all())
    max_d = float((both["delta_0"] - both["delta_1"]).abs().max()) if len(both) else 0.0
    n_neq = int((both["delta_0"].to_numpy() != both["delta_1"].to_numpy()).sum())
    # report 全等: 双方均经 json 往返（default=str 兜底 numpy 标量），逐键精确比较
    rep0_cmp = json.loads((OUT / "snapshot_report.json").read_text())
    rep1_cmp = json.loads(json.dumps(report, default=str))
    report_identical = rep0_cmp == rep1_cmp
    res = {"phase": "compare", "rows_snapshot": len(snap), "rows_new": len(real),
           "n_both": n_both, "only_snapshot": only0, "only_new": only1,
           "max_abs_delta": max_d, "n_bitwise_neq": n_neq, "no_nan": no_nan,
           "report_identical": report_identical,
           "snapshot_sha256_match": snap_ok}
    res["pass"] = (max_d == 0.0 and only0 == 0 and only1 == 0 and len(snap) == len(real)
                   and no_nan and report_identical)
    (OUT / "compare_result.json").write_text(json.dumps(res, indent=2))
    print(f"[compare] rows {len(snap)}=={len(real)}? {len(snap) == len(real)}; "
          f"max|delta|={max_d}; only0={only0} only1={only1}; report_identical={rep0_cmp == rep1_cmp}")
    print(f"[compare] PASS={res['pass']}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
