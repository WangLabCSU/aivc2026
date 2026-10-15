#!/usr/bin/env python
"""C1 冻结资产复现断言脚本（v2.2 fid/源质量加权线，REV C1）。

对两份冻结 csv 从 §3.1 配方全量重算并断言：
  1) module_spectra_z_frozen.csv —— 10 模块 × 9 上下文模块 z 谱，逐位相等（maxdiff=0.0, np.array_equal）；
  2) rho_9ctx_frozen.csv        —— 9×9 上下文×上下文 Spearman ρ 矩阵，最大差 ≤ 1.11e-16（1 ULP，机器精度）。

配方（冻结，勿改；与 docs/team/cb_v22_c1_freeze.md §3.1 逐字一致）：
  - 词表 data/interim/l15_fusion/v3/vocab.json（14,372），v2i = {g: i}
  - ctxs = [a,b,c,hepg2,jurkat,rpe1,hct116,gwps,k562_ess]
  - M = np.stack([basal[s] for s in ctxs]) → (9, 14372) float32 上下文主轴
  - mu = M.mean(0); sd = M.std(0)（ddof=0）; sd_safe = np.where(sd<1e-8, 1.0, sd); Z = (M-mu)/sd_safe
  - 模块值 vals = Z[:, idx].mean(1)，idx = [v2i[g] for g in genes if g in v2i]（in-vocab 子集均值）
  - ρ = scipy.stats.spearmanr(谱_a, 谱_b)，n=10 模块点
冻结依据：docs/team/cb_v22_c1_freeze.md（C1: LOPO 执行前冻结，冻结后不得因读数调整 ρ/τ/β）。
运行：source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026 && \
     python scripts/repro_fid_srcquality_frozen.py
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = "/data/resource/wanglab/aivc/aivc2026"
V3 = f"{ROOT}/data/interim/l15_fusion/v3"
FROZEN_DIR = f"{ROOT}/data/interim/fid_srcquality"
SPECTRA_CSV = f"{FROZEN_DIR}/module_spectra_z_frozen.csv"
RHO_CSV = f"{FROZEN_DIR}/rho_9ctx_frozen.csv"

CTXS = ["a", "b", "c", "hepg2", "jurkat", "rpe1", "hct116", "gwps", "k562_ess"]
MODULES = ["P53_axis", "IFN", "E2F_Prolif", "MYC", "EMT_mes", "Epithelial",
           "Myeloid", "Erythroid", "Tcell", "Endothelial"]

# 十模块基因列表（C1 冻结文档 §2 逐字；勿改）
MODS10 = {
    "P53_axis": "CDKN1A BBC3 MDM2 TP53 GADD45A GADD45B DDB2 SESN1 SESN2 RRM2B PMAIP1 BAX PERP ZMAT3 TRIAP1 PLK3 EI24".split(),
    "IFN": "ISG15 IFIT1 IFIT2 IFIT3 IFI6 IFI44 IFI44L IFITM1 IFITM3 OAS1 OAS2 OASL MX1 MX2 IRF7 IRF1 RSAD2 DDX58 IFIH1 CXCL10 STAT1 STAT2".split(),
    "E2F_Prolif": "MKI67 TOP2A PCNA MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 E2F1 E2F2 AURKA BIRC5 CCNB1 CCNE1 TYMS FEN1 RRM2 UBE2C CDC20 CENPF HMGB2 NUSAP1 PRC1 ASPM".split(),
    "MYC": "MYC NPM1 ODC1 CAD LDHA PDK1 SLC1A5 GLS CDK4 TERT".split(),
    "EMT_mes": "VIM FN1 CDH2 SNAI1 SNAI2 TWIST1 ZEB1 ZEB2 ITGA5 ITGB1 SPARC SERPINE1 COL1A1".split(),
    "Epithelial": "EPCAM KRT8 KRT18 KRT19 KRT7 CDH1 CLDN3 CLDN4 CLDN7 TJP1 DSP".split(),
    "Myeloid": "CD68 CSF1R TYROBP CST3 CTSS MPO LST1 HCST".split(),
    "Erythroid": "ALAS2 EPOR FECH GYPA KLF1 HBA1 TAL1".split(),
    "Tcell": "CD3D CD3E CD3G TRAC CD2 CD5 CD7 CD28 LCK IL7R".split(),
    "Endothelial": "PECAM1 FLT1 ICAM2 TEK".split(),
}


def main() -> int:
    vocab = json.load(open(f"{V3}/vocab.json"))
    vocab = list(vocab) if isinstance(vocab, list) else list(vocab.keys())
    v2i = {g: i for i, g in enumerate(vocab)}

    basal = {s: np.load(f"{V3}/basal_{s}.npy") for s in CTXS}
    M = np.stack([basal[s] for s in CTXS])          # (9, 14372) ctx-major
    mu, sd = M.mean(0), M.std(0)                    # ddof=0
    sd_safe = np.where(sd < 1e-8, 1.0, sd)
    Z = (M - mu) / sd_safe                          # (9, 14372)

    # --- 1) 模块谱逐位断言 ---
    spec_frozen = pd.read_csv(SPECTRA_CSV, index_col=0)
    assert list(spec_frozen.columns) == CTXS, f"spectra 列序漂移: {list(spec_frozen.columns)}"
    assert list(spec_frozen.index) == MODULES, f"spectra 行序漂移: {list(spec_frozen.index)}"

    spec_recomputed = {}
    in_vocab_counts = {}
    for m in MODULES:
        genes = MODS10[m]
        idx = [v2i[g] for g in genes if g in v2i]
        in_vocab_counts[m] = (len(idx), len(genes))
        vals = Z[:, idx].mean(1)
        spec_recomputed[m] = vals.astype(np.float32)

    n_exact, max_diff_spec = 0, 0.0
    for m in MODULES:
        frozen_vec = spec_frozen.loc[m].values.astype(np.float32)
        rec_vec = spec_recomputed[m]
        exact = np.array_equal(rec_vec, frozen_vec)
        d = float(np.abs(rec_vec.astype(np.float64) - frozen_vec.astype(np.float64)).max())
        n_exact += int(exact)
        max_diff_spec = max(max_diff_spec, d)
        print(f"  {m:<12} in-vocab {in_vocab_counts[m][0]}/{in_vocab_counts[m][1]}  "
              f"bit-exact={exact}  maxdiff={d:.3e}")
    print(f"[1/2] module spectra: bit-exact {n_exact}/10, global maxdiff={max_diff_spec:.3e}")
    assert n_exact == 10, f"模块谱逐位复现失败: {10 - n_exact} 个模块有偏差"
    assert max_diff_spec == 0.0, "maxdiff 应为 0"

    # --- 2) ρ 矩阵重算断言（≤1.11e-16, 1 ULP）---
    rho_frozen = pd.read_csv(RHO_CSV, index_col=0)
    assert list(rho_frozen.columns) == CTXS, f"rho 列序漂移: {list(rho_frozen.columns)}"
    assert list(rho_frozen.index) == CTXS, f"ρ 行序漂移: {list(rho_frozen.index)}"

    spec_df = pd.DataFrame(spec_recomputed, index=CTXS).T   # (10, 9)
    rho_rec = pd.DataFrame(index=CTXS, columns=CTXS, dtype=float)
    for s1 in CTXS:
        for s2 in CTXS:
            r = spearmanr(spec_df[s1].values, spec_df[s2].values).statistic
            rho_rec.loc[s1, s2] = float(r)

    diff = np.abs(rho_rec.values.astype(np.float64) - rho_frozen.values.astype(np.float64))
    max_diff_rho = float(diff.max())
    ulp = 2.0 ** -53   # = 1.1102230246251565e-16，冻结文档 §3.2 所述 "1.11e-16 (1 ULP)" 的精确值
    print(f"[2/2] rho matrix recompute: maxdiff = {max_diff_rho:.3e} (tolerance {ulp:.3e} = 1 ULP)")
    assert max_diff_rho <= ulp, f"ρ 重算超差: {max_diff_rho:.3e} > {ulp:.3e}"

    print("REPRO PASS: 两冻结 csv 均复现（spectra bit-exact 10/10; rho ≤ 1.11e-16）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
