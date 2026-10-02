"""HCT116 全深度 fid 检验：聚合归一化/原始分数 → hct116_fulldepth.json + summary 追加 + commit。

用法: python scripts/hct116_fulldepth_aggregate.py [--commit]
幂等: 重复运行覆盖同名的 json 与 md 自动小节（marker 之间）。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
OUT = ROOT / "outputs/metrics/local_bench"
MD = OUT / "fusion_v2_summary.md"
MD_BEGIN = "<!-- hct116-fulldepth:auto-begin -->"
MD_END = "<!-- hct116-fulldepth:auto-end -->"
TAGS = {
    "heuristic_lam15": "l11_lam1.5",
    "fusionv1_lam25": "l15v1_hct116_lam2.5",
    "fusionv2_lam15": "l15v2_hct116_lam1.5",
    "fusionv2_lam25": "l15v2_hct116_lam2.5",
}
CTX = "S_HCT116"


def ref_stats() -> dict:
    import anndata as ad
    ref = OUT.parent.parent.parent / "data/interim/local_bench/S_HCT116/ref.h5ad"
    a = ad.read_h5ad(ref, backed="r")
    vc = a.obs["target_gene"].astype(str).value_counts()
    ctrl = "Non-Targeting" if "Non-Targeting" in vc.index else "non-targeting"
    st = {"shape": list(a.shape),
          "n_targets": int(len(vc) - (1 if ctrl in vc.index else 0)),
          "ntc_cells": int(vc.get(ctrl, 0)),
          "median_cells_per_target": float(vc.drop(index=[ctrl], errors="ignore").median())}
    a.file.close()
    return st


def raw_metrics(tag: str) -> dict:
    p = OUT / tag / f"{CTX}_eval" / "agg_results.csv"
    if not p.exists():
        return {}
    raw = pd.read_csv(p)
    row = raw[raw["statistic"] == "mean"].iloc[0]
    return {k.split("de_wilcoxon_")[-1].split("expr_")[-1]: (None if pd.isna(row[k]) else float(row[k]))
            for k in raw.columns if k not in ("statistic",) and not pd.isna(row[k])
            and k != "statistic"} if len(row) else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()
    norm = json.loads((OUT / "hct116_fulldepth_norm.json").read_text())["per_tag"]
    pooled = json.loads((ROOT / "data/public/xatlas/.pooled_full.done").read_text())
    ref = ref_stats()

    per_tag = {}
    for key, tag in TAGS.items():
        n = norm.get(tag, {}).get("per_context", {}).get(CTX, {})
        per_tag[key] = {
            "tag": tag,
            "norm": n.get("per_metric", {}),
            "overall_norm": n.get("overall"),
            "bundle_mode": n.get("bundle_mode"),
            "metrics_unavailable": n.get("metrics_unavailable", []),
            "raw": raw_metrics(tag),
        }

    fid_h = per_tag["heuristic_lam15"]["norm"].get("fid")
    fid_v2_15 = per_tag["fusionv2_lam15"]["norm"].get("fid")
    fid_v2_25 = per_tag["fusionv2_lam25"]["norm"].get("fid")
    fid_v1 = per_tag["fusionv1_lam25"]["norm"].get("fid")
    comparisons = {
        "fid_fusionv2_lam15_vs_heuristic_lam15": (
            None if fid_v2_15 is None or fid_h is None else round(fid_v2_15 - fid_h, 4)),
        "fid_fusionv2_lam25_vs_heuristic_lam15": (
            None if fid_v2_25 is None or fid_h is None else round(fid_v2_25 - fid_h, 4)),
        "fid_fusionv1_lam25_vs_heuristic_lam15": (
            None if fid_v1 is None or fid_h is None else round(fid_v1 - fid_h, 4)),
        "fusion_fixes_fid_on_p53wt_crc": (
            None if fid_v2_15 is None or fid_h is None else bool(fid_v2_15 - fid_h > 0.05)),
    }
    unavail = sorted({m for v in per_tag.values() for m in v["metrics_unavailable"]})
    result = {
        "scale": "manual/bundle anchor-normalized (0=baseline, 1=replicate-halfsplit proxy)",
        "pooled": pooled, "ref": ref,
        "nmae_disposition": ("unavailable (degenerate anchor at depth / gate-empty perts)"
                             if "nmae" in unavail else f"available (unavailable: {unavail})"),
        "per_tag": per_tag, "comparisons": comparisons,
        "notes": [
            "S_HCT116 ref 由全批次 pooled (min_cells 30 gate) 重建; ref 靶/NTC 块 400/2000 细胞帽",
            "fusion v2 对 S_HCT116 为 in-sample (hct116 在训练池; sig_hct116 特征按 row_ctx 约定置 NaN)",
            "fusion v1 (v1h) 为 LOCO-correct 消融: 训练池不含 hct116",
            "启发式对照 = l11_lam1.5 同 ref 重跑 (ws-v1.2-sampled-lam15, λ=1.5)",
        ],
    }
    (OUT / "hct116_fulldepth.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(comparisons, indent=1))

    # ---- summary md 追加（幂等替换 auto 小节）
    def fmt(v, d=4):
        return "n/a" if v is None else f"{v:.{d}f}"
    rows = ""
    for key, v in per_tag.items():
        nm = v["norm"]
        rows += (f"| {key} | {fmt(nm.get('fid'))} | {fmt(nm.get('pds'))} "
                 f"| {fmt(nm.get('reach'))} | {fmt(nm.get('jac'))} | {fmt(nm.get('nmae'))} "
                 f"| {fmt(v['overall_norm'])} | {v['bundle_mode'] or '-'} |\n")
    section = f"""
{MD_BEGIN}
## 附: HCT116 全深度 fid 坍塌检验（夜间自动链, S_HCT116 全批次 ref）

- pooled: {pooled['n_cells']} cells / {pooled['n_batches']} batches / NTC {pooled['n_ntc']} /
  ≥30-cell perts {pooled['perts_ge30']} (median {pooled['median_cells_per_pert']});
  ref: {ref['shape'][0]} cells, {ref['n_targets']} targets (median {ref['median_cells_per_target']} cells/靶)
- 归一化 fid 对比: **fusion v2@λ1.5 vs 启发式 λ1.5 = {fmt(fid_v2_15)} vs {fmt(fid_h)}
  (Δ={fmt(comparisons['fid_fusionv2_lam15_vs_heuristic_lam15'])})**;
  fusion v2@λ2.5 = {fmt(fid_v2_25)}; fusion v1(LOCO)@λ2.5 = {fmt(fid_v1)}
- 结论: {"fusion 在 p53-WT CRC 上下文复现 fid 修复" if comparisons["fusion_fixes_fid_on_p53wt_crc"] else "fusion fid 修复在本上下文未达 +0.05 判据（见 json comparisons）"}
- nmae 处置: {result['nmae_disposition']}
- 官方 -0.838 型坍塌是否复现: 见启发式行 fid/overall 与 Q_HepG2(0.73/0.27) 对照

| 臂 | fid | pds | reach | jac | nmae | overall | anchor |
|---|---|---|---|---|---|---|---|
{rows}{MD_END}
"""
    if MD.exists():
        text = MD.read_text()
        if MD_BEGIN in text:
            text = re.sub(re.escape(MD_BEGIN) + r".*?" + re.escape(MD_END), section.strip(),
                          text, flags=re.S)
        else:
            text = text.rstrip() + "\n" + section
        MD.write_text(text)
    else:
        MD.write_text("# fusion v2 汇总\n" + section)

    if args.commit:
        subprocess.run(["git", "add", "-A", "src/", "configs/", "scripts/"], check=True)
        subprocess.run(["git", "add", "-f", str(MD.relative_to(ROOT)),
                        str((OUT / "hct116_fulldepth.json").relative_to(ROOT)),
                        str((OUT / "hct116_fulldepth_norm.json").relative_to(ROOT))], check=True)
        msg = ("bench: HCT116 全深度 fid 检验——pooled 全批次重建+幂等链, "
               f"fusion v2 vs 启发式 fid {fmt(fid_v2_15)}/{fmt(fid_h)} "
               f"(Δ={fmt(comparisons['fid_fusionv2_lam15_vs_heuristic_lam15'])}), "
               f"nmae={'unavailable' if 'nmae' in unavail else 'ok'}")
        r = subprocess.run(["git", "commit", "-m", msg], capture_output=True, text=True)
        print(r.stdout or r.stderr)
    print("[aggregate] done")


if __name__ == "__main__":
    main()
