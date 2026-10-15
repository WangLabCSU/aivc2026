# gene_align/DECLARATION.md — v0.6 Option a 基因空间对齐声明

- 日期: 2026-10-15; 撰写: ml-impl-v06 (scripts/v06_build_gene_align.py 生成, PI 裁定 B=Option a / C=Option a)
- 服务对象: bizard-v0.6 DL Δ̂ 训练 (`docs/team/rev_prereg_bizard_v06_dl_delta.md` §2/§4)

## 源表

| 表 | 来源 | 维度 |
|---|---|---|
| 竞赛 panel | `data/raw/controls/gene_names.csv` | 18533 基因 (vcc2026-val-1 manifest n_genes=18533) |
| st2000 HVG | `data/public/models/ST-HVG-Replogle/*/var_dims.pkl` 键 `gene_names` (8 官方 ckpt 四上下文逐位相同, outputs/v06_alignment_audit/audit.md §A) | 2000 |
| se600m HVG | `data/interim/state_retrain/run_v2/state_se600m_gwps_rpe1/var_dims.pkl` 键 `gene_names` | 8192 |

## 交集率

- st2000 ∩ panel = 1877/2000 = 0.9385 (与审计 1877/2000=93.85% 一致)
- se600m ∩ panel = 7786/8192 = 0.9504 (与审计 7786/8192=95.04% 一致)
- 两族 HVG 交集 1936 个 (st2000∩se600m=1936, audit §A)

## 映射语义 (Option a, 零填充规则)

- 编码器输入 = ckpt 原生 HVG 排列, 顺序即 `mapping_{family}.csv` 的 `hvg_pos` 列, **不可重排**;
- panel 基因向量投影到 ckpt 空间: `in_panel=True` 的位置取该基因真实表达, `in_panel=False`(panel_idx=-1)的位置**置零**;
- encoder 主干权重完整继承 (输入维度与排列与 ckpt 训练时一致);
- 集外基因信息经 decoder/输出头 (输出空间重训为全 panel 18533) 补足, 扰动 onehot 表按 PI 裁定重建 (9898 项, pert 嵌入权重不继承);
- k562_ess 第六上下文复用 st2000 同一张表 (同细胞系 k562, PI 裁定 C)。

## 文件

- `mapping_st2000.csv`: hvg_gene, hvg_pos, panel_idx(-1=零填充), in_panel — 2000 行
- `mapping_se600m.csv`: 同 schema — 8192 行

spot-check (10 基因, 全部 OK):
```
  HES4 -> 15417  OK panel[15417]=HES4
  ISG15 -> 15275  OK panel[15275]=ISG15
  MIB2 -> 16034  OK panel[16034]=MIB2
  CEP104 -> 4401  OK panel[4401]=CEP104
  ACOT7 -> 2008  OK panel[2008]=ACOT7
  HMGN2 -> -1  OK
  SNHG3 -> -1  OK
  SNHG12 -> -1  OK
  MRPS15 -> -1  OK
  RPS8 -> -1  OK
```
