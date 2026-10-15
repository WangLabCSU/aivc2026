#!/usr/bin/env python
"""v2.2 ②source_context_weights.parquet 构建脚本（fid/源质量加权线，REV C1 后按 PI 放行顺序执行）。

产出 data/interim/fid_srcquality/source_context_weights.parquet：
  - schema: (source, source_context, context, rho_raw, rho_used) + 模块谱快照列
    src_z_<module> × 10 + ctx_z_<module> × 10（C1 要求的"计算用的模块谱快照列"）；
  - 源→basal 上下文映射（冻结文档 §4）：gwps→gwps、hct116_all→hct116、jurkat_all→jurkat、
    k562_ess_all→k562_ess、rpe1_all→rpe1、hepg2_all→hepg2；norman_all 不指派 ρ（NaN 行占位）；
  - rho_used = max(rho_raw, 0)（冻结 §5，永不翻号）；
  - 目标上下文覆盖全部 9 ctx（LOPO 用 hepg2/rpe1，共识空间用 a/b/c，全部在档）。

设计决策（写入 build_meta，随 ⑤ 纪要送 REV）：
  本线不另发模型侧合并权重表（[source, gene, w]）。原因：make_predictions.py 的 gene_weights
  键为 (source, gene) 且 gene-only 键合——无 target/context 维度，A.1 的逐单元 w（2·agree−1，
  β 收缩）与逐 (source, context) ρ 均无法经该键表达；逐源常数 w 退化为 dataset_weights
  （等价于闭环 FAIL 的 C3 全局臂权重形态）。逐单元置信权重按 spec A.3 交付于
  consensus_delta.parquet 的 w 列（LOPO PASS 分支）。

运行：source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026 && \
     python scripts/build_fid_srcquality_02_weights.py
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = "/data/resource/wanglab/aivc/aivc2026"
FQ = f"{ROOT}/data/interim/fid_srcquality"
SPECTRA_CSV = f"{FQ}/module_spectra_z_frozen.csv"
RHO_CSV = f"{FQ}/rho_9ctx_frozen.csv"
OUT_PARQUET = f"{FQ}/source_context_weights.parquet"
OUT_META = f"{FQ}/source_context_weights_build_meta.json"

CTXS = ["a", "b", "c", "hepg2", "jurkat", "rpe1", "hct116", "gwps", "k562_ess"]
MODULES = ["P53_axis", "IFN", "E2F_Prolif", "MYC", "EMT_mes", "Epithelial",
           "Myeloid", "Erythroid", "Tcell", "Endothelial"]

# 冻结文档 §4 映射（源名 = 签名 parquet 文件名去后缀；冻结文档 §4 中 "hepg2" 即 hepg2_all.parquet）：
# 源签名 parquet → basal 上下文；norman_all 不指派 ρ
SRC2CTX = {
    "gwps": "gwps",
    "hct116_all": "hct116",
    "hepg2_all": "hepg2",
    "jurkat_all": "jurkat",
    "k562_ess_all": "k562_ess",
    "rpe1_all": "rpe1",
    "norman_all": None,   # C2 剔出投票池 + 无 basal 归属 → ρ 不指派
}

ULP = 2.0 ** -53   # 1.1102230246251565e-16


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    spec = pd.read_csv(SPECTRA_CSV, index_col=0)   # (10 modules, 9 ctxs) float32
    rho = pd.read_csv(RHO_CSV, index_col=0)        # (9, 9) float64
    assert list(spec.columns) == CTXS and list(spec.index) == MODULES
    assert list(rho.index) == CTXS and list(rho.columns) == CTXS

    rows = []
    for src, sctx in SRC2CTX.items():
        for tgt in CTXS:
            if sctx is None:
                row = {"source": src, "source_context": "none", "context": tgt,
                       "rho_raw": np.nan, "rho_used": np.nan}
                for m in MODULES:
                    row[f"src_z_{m}"] = np.float32(np.nan)
                    row[f"ctx_z_{m}"] = np.float32(spec.loc[m, tgt])
            else:
                rho_raw = float(rho.loc[sctx, tgt])
                row = {"source": src, "source_context": sctx, "context": tgt,
                       "rho_raw": rho_raw, "rho_used": max(rho_raw, 0.0)}
                for m in MODULES:
                    row[f"src_z_{m}"] = np.float32(spec.loc[m, sctx])
                    row[f"ctx_z_{m}"] = np.float32(spec.loc[m, tgt])
            rows.append(row)

    df = pd.DataFrame(rows)
    # 列序规范化 + 快照列显式 float32
    zcols = [f"src_z_{m}" for m in MODULES] + [f"ctx_z_{m}" for m in MODULES]
    df = df[["source", "source_context", "context", "rho_raw", "rho_used"] + zcols]
    for c in zcols:
        df[c] = df[c].astype(np.float32)

    # 断言 1：快照列重算 ρ 与冻结矩阵一致（≤1 ULP）
    max_diff = 0.0
    for _, r in df.iterrows():
        if r["source_context"] == "none":
            continue
        sv = np.array([r[f"src_z_{m}"] for m in MODULES], dtype=np.float32)
        tv = np.array([r[f"ctx_z_{m}"] for m in MODULES], dtype=np.float32)
        r_rec = float(spearmanr(sv, tv).statistic)
        max_diff = max(max_diff, abs(r_rec - float(r["rho_raw"])))
    assert max_diff <= ULP, f"快照重算 ρ 超差: {max_diff:.3e} > {ULP:.3e}"
    print(f"[check] 快照列 spearmanr 重算 vs rho_raw: maxdiff = {max_diff:.3e} (≤1 ULP)")

    # 断言 2：rho_used 规则 + norman 不指派
    m = df["rho_raw"].notna()
    assert np.array_equal(df.loc[m, "rho_used"].values, np.maximum(df.loc[m, "rho_raw"].values, 0.0))
    assert df.loc[~m, "rho_used"].isna().all()
    assert set(df.loc[m, "source"]) == {k for k, v in SRC2CTX.items() if v}
    assert df.loc[df["source"] == "norman_all", "rho_raw"].isna().all()
    print(f"[check] rho_used = max(rho_raw, 0) 全 {int(m.sum())} 行成立；"
          f"norman_all {len(df) - int(m.sum())} 行 NaN（不指派）")

    # 断言 3：ρ 对称读数抽检（冻结矩阵关键读数）
    def rr(s, t):
        return float(df.loc[(df.source == {"gwps": "gwps", "hct116_all": "hct116_all"}[s]) &
                            (df.context == t), "rho_raw"].iloc[0])
    assert abs(rr("gwps", "k562_ess") - 0.9515151515151515) < 1e-12
    assert abs(rr("hct116_all", "rpe1") - 0.5515151515151515) < 1e-12
    assert abs(rr("hct116_all", "hepg2") - 0.6848484848484848) < 1e-12
    print("[check] 冻结关键读数抽检通过（gwps↔k562_ess 0.9515 / hct116↔rpe1 0.5515 / hct116↔hepg2 0.6848）")

    df.to_parquet(OUT_PARQUET, index=False)

    meta = {
        "built": "2026-10-14",
        "builder": "cb-expert",
        "script": "scripts/build_fid_srcquality_02_weights.py",
        "frozen_refs": {
            "module_spectra_z_frozen.csv.sha256": sha256_of(SPECTRA_CSV),
            "rho_9ctx_frozen.csv.sha256": sha256_of(RHO_CSV),
        },
        "source_to_context_mapping": {k: (v if v else "none(no rho assigned)") for k, v in SRC2CTX.items()},
        "naming_note": "source 名 = 签名 parquet 文件名去 .parquet；冻结文档 §4 的 hepg2 即 hepg2_all.parquet",
        "rho_definition": "rho[source_context, context] = Spearman(module z spectra, n=10); scipy.stats.spearmanr",
        "rho_used_rule": "rho_used = max(rho_raw, 0); 永不翻号; norman_all 不指派 ρ（NaN 行占位）",
        "snapshot_columns": zcols,
        "rows": len(df),
        "output_sha256": None,
        "model_facing_weight_table_decision": {
            "merged_table_emitted": False,
            "rationale": ("make_predictions.py gene_weights 键为 (source, gene) 且 gene-only 键合，"
                          "无 target/context 维——A.1 逐单元 w（2*agree-1, β 收缩）与逐 (source,context) ρ "
                          "均无法经该键表达；逐源常数 w 退化为 dataset_weights（等价闭环 FAIL 的 C3 全局臂权重形态）。"
                          "逐单元置信 w 按 spec A.3 交付于 consensus_delta.parquet 的 w 列（LOPO PASS 分支）；"
                          "gene_weights 键路径保持可用但本线不使用。"),
        },
    }
    meta["output_sha256"] = sha256_of(OUT_PARQUET)
    with open(OUT_META, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[write] {OUT_PARQUET} rows={len(df)} cols={len(df.columns)}")
    print(f"[check] output sha256 = {meta['output_sha256']}")
    print("BUILD 02 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
