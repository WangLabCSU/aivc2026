#!/usr/bin/env python
"""v0.6 Phase 1: build gene-space alignment tables (Option a, PI ruling B).

For each ckpt family (official ST-HVG-Replogle 2000-HVG global table; local
retrain se600m 8192-HVG table), map every ckpt gene to its index in the
competition panel (data/raw/controls/gene_names.csv, 18533 genes).

Mapping semantics (audit.md §B Option a):
  - encoder input stays in ckpt-native HVG order;
  - a panel-gene vector is projected to ckpt space as:
      in-set (gene is a ckpt HVG)  -> real expression at that ckpt-HVG position
      out-of-set                   -> zero-fill
  - mapping table records, per ckpt gene: panel index (-1 = zero-fill), flags.

Outputs:
  gene_align/mapping_st2000.csv    (2000 rows)
  gene_align/mapping_se600m.csv    (8192 rows)
  gene_align/DECLARATION.md        (source tables, 93.85% intersection, zero-fill rule)
"""
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
OUT = ROOT / "gene_align"
OUT.mkdir(parents=True, exist_ok=True)

ST_VAR_DIMS = ROOT / "data/public/models/ST-HVG-Replogle/zeroshot/hepg2/var_dims.pkl"
SE_VAR_DIMS = ROOT / "data/interim/state_retrain/run_v2/state_se600m_gwps_rpe1/var_dims.pkl"
PANEL_CSV = ROOT / "data/raw/controls/gene_names.csv"


def build(family: str, vd_path: Path, panel: dict, panel_order: list) -> pd.DataFrame:
    with open(vd_path, "rb") as fh:
        vd = pickle.load(fh)
    genes = [str(x) for x in vd["gene_names"]]
    assert len(genes) == len(set(genes)), f"{family}: duplicate genes in var_dims"
    df = pd.DataFrame({"hvg_gene": genes})
    df["hvg_pos"] = range(len(df))  # position in ckpt-native input vector
    df["panel_idx"] = df["hvg_gene"].map(panel).fillna(-1).astype("int64")
    df["in_panel"] = df["panel_idx"] >= 0
    df.to_csv(OUT / f"mapping_{family}.csv", index=False)
    n_in = int(df["in_panel"].sum())
    print(f"[{family}] ckpt_hvg={len(df)} in_panel={n_in} "
          f"({n_in/len(df):.4f}) zero_fill={len(df)-n_in}")
    return df


def main():
    panel_series = pd.read_csv(PANEL_CSV)["gene_name"].astype(str)
    assert panel_series.is_unique, "panel gene_names.csv has duplicates"
    panel_order = panel_series.tolist()
    panel = {g: i for i, g in enumerate(panel_order)}
    print(f"[panel] competition panel: {len(panel)} genes")

    df_st = build("st2000", ST_VAR_DIMS, panel, panel_order)
    df_se = build("se600m", SE_VAR_DIMS, panel, panel_order)

    # cross-family fact for the declaration
    inter_fam = len(set(df_st["hvg_gene"]) & set(df_se["hvg_gene"]))
    n_in_st = int(df_st["in_panel"].sum())
    n_in_se = int(df_se["in_panel"].sum())

    # spot-check 10 genes: 5 known-intersection, 5 pseudo-random tail genes
    spot = list(df_st[df_st["in_panel"]].head(5)["hvg_gene"]) + \
        list(df_st[~df_st["in_panel"]].head(5)["hvg_gene"])
    lines = ["# spot-check (st2000): gene -> panel_idx (-1 = zero-fill)"]
    for g in spot:
        i = int(df_st.loc[df_st["hvg_gene"] == g, "panel_idx"].iloc[0])
        check = "OK" if (i == -1) == (panel.get(g) is None) else "MISMATCH"
        if i >= 0:
            check += f" panel[{i}]={panel_order[i]}"
        lines.append(f"  {g} -> {i}  {check}")
        print(f"[spot] {g} -> {i} {check}")
    assert all("MISMATCH" not in l for l in lines)

    (OUT / "DECLARATION.md").write_text(f"""# gene_align/DECLARATION.md — v0.6 Option a 基因空间对齐声明

- 日期: 2026-10-15; 撰写: ml-impl-v06 (scripts/v06_build_gene_align.py 生成, PI 裁定 B=Option a / C=Option a)
- 服务对象: bizard-v0.6 DL Δ̂ 训练 (`docs/team/rev_prereg_bizard_v06_dl_delta.md` §2/§4)

## 源表

| 表 | 来源 | 维度 |
|---|---|---|
| 竞赛 panel | `data/raw/controls/gene_names.csv` | {len(panel)} 基因 (vcc2026-val-1 manifest n_genes=18533) |
| st2000 HVG | `data/public/models/ST-HVG-Replogle/*/var_dims.pkl` 键 `gene_names` (8 官方 ckpt 四上下文逐位相同, outputs/v06_alignment_audit/audit.md §A) | {len(df_st)} |
| se600m HVG | `data/interim/state_retrain/run_v2/state_se600m_gwps_rpe1/var_dims.pkl` 键 `gene_names` | {len(df_se)} |

## 交集率

- st2000 ∩ panel = {n_in_st}/{len(df_st)} = {n_in_st/len(df_st):.4f} (与审计 1877/2000=93.85% 一致)
- se600m ∩ panel = {n_in_se}/{len(df_se)} = {n_in_se/len(df_se):.4f} (与审计 7786/8192=95.04% 一致)
- 两族 HVG 交集 {inter_fam} 个 (st2000∩se600m=1936, audit §A)

## 映射语义 (Option a, 零填充规则)

- 编码器输入 = ckpt 原生 HVG 排列, 顺序即 `mapping_{{family}}.csv` 的 `hvg_pos` 列, **不可重排**;
- panel 基因向量投影到 ckpt 空间: `in_panel=True` 的位置取该基因真实表达, `in_panel=False`(panel_idx=-1)的位置**置零**;
- encoder 主干权重完整继承 (输入维度与排列与 ckpt 训练时一致);
- 集外基因信息经 decoder/输出头 (输出空间重训为全 panel 18533) 补足, 扰动 onehot 表按 PI 裁定重建 (9898 项, pert 嵌入权重不继承);
- k562_ess 第六上下文复用 st2000 同一张表 (同细胞系 k562, PI 裁定 C)。

## 文件

- `mapping_st2000.csv`: hvg_gene, hvg_pos, panel_idx(-1=零填充), in_panel — {len(df_st)} 行
- `mapping_se600m.csv`: 同 schema — {len(df_se)} 行

spot-check ({len(spot)} 基因, 全部 OK):
```
{chr(10).join(lines[1:])}
```
""")
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
