# scbasecount 特征化报告（CB，2026-10-14）

> 承接任务：对已修复的 10 个 h5ad 逐研究核实扰动标签，对可靠子集按
> (target, cell_line) 伪批量 logFC vs NTC（build_orion_signatures_v3.py 口径）特征化。
> **结论先行：特征化被阻断——所有 10 个研究的 h5ad 快照均不含 per-cell guide 分配，
> 无法按靶拆分池化筛选细胞，per-target 伪批量签名无法构造。**

## 一、逐研究扰动核实（sample_metadata + h5ad 结构 + 文献设计三方对证）

| 研究 | SRX | 细胞数 | 实际扰动设计 | per-cell guide 列 | 可按靶拆分 |
|------|-----|--------|--------------|-------------------|-----------|
| hek293ft | SRX24879306 / 07 | 113,914 / 69,064 | **池化** 42 基因 CPA（剪切与多腺苷酸化）CRISPR 筛选，feature_barcoding 建库 | 无 | 否 |
| telohaec | SRX22713290 / 98 | 86,389 / 92,480 | doxy 诱导 CRISPRi Perturb-seq，**36,880-guide 文库**（两样本为同一文库的不同诱导臂） | 无 | 否 |
| aics | SRX22915751 / 53 | 98,299 / 82,977 | iPSC（AICS-0090 cl.391）**87 个 ASD 基因** CRISPRi（混合池） | 无 | 否 |
| mdamb361/231 | SRX19325980/90/96/97 | 33,629 / 85,439 / 54,704 / 54,504 | **增强子靶向** CRISPRi（非基因靶，target 也不是 300 靶口径） | 无 | 否（且不适用） |

核实方法：
1. h5ad 逐文件穷举审计 obs/obsm/uns：10/10 全部为 obs=[SRX_accession, _index,
   cell_ontology_term_id, cell_type, gene/umi_count×3 层]，obsm=[]、uns=[]；
   var 仅 36,601 个基因（含 CPA 研究 feature_barcoding 建库样本），**无 guide feature 条目**。
2. obs_metadata.parquet（6.07GB，3.02 亿行）列=cell_barcode, SRX_accession,
   gene/umi_count×3, cell_type, cell_ontology_term_id——**无 perturbation 列**。
   barcode 联动已验证可用（86,389 barcode 全集匹配），但联动产出只有 (SRX, cell_type)。
3. GCS 桶 2026-01-12 快照目录实测：h5ad 模式仅 GeneFull / GeneFull_ExonOverIntron /
   Gene / Velocyto 四种——**全库不含 guide-capture（CRISPR feature）模式**，
   guide 分配在本快照层面不存在，不是"没下载到"而是"快照不携带"。

## 二、阻断机制说明

四个研究全部是池化（pooled）CRISPR 筛选：单个 SRX 内的细胞混合了数十至上万个
guide 的扰动。签名构造要求 (细胞 → 靶) 的逐细胞映射，该映射只存在于原作者的
guide 分配表（GEO 级原始处理产物），scBaseCount 快照不携带。因此：

- per-target 伪批量 logFC vs NTC **不可计算**（无法分组）；
- 退而求其次的 SRX 级"整池 logFC"是数十/上万个扰动的平均混合效应，
  与 300 靶签名库的 per-target 语义不兼容，不产出。

`data/interim/scbasecount_features/` 目录保留为空（已创建），**无签名产出**。

## 三、与竞赛 300 靶的交集（仅候选性，不可用）

若 guide 分配可得，理论可触达 5/300：**PAPOLA**（hek293ft CPA 42 基因筛选）、
**ADNP / CHD3 / DEAF1 / KMT2A**（aics 87 ASD 基因 CRISPRi）。
mdamb 增强子筛选与 telohaec 文库设计均不贡献基因靶。
现状：这 5 个候选全部因无 per-cell guide 分配而**不可用**。

## 四、对 v2.0 特征源使用建议

1. **scBaseCount 不作签名扩源**（与 9/30 方案第 6 节结论一致且加强）：
   本地 10 研究均为池化筛选且快照无 guide 列；元数据 perturbation 字段
   'CRISPRi'/'pooled single-cell CRISPR screen...' 只描述建库设计，不提供靶分配。
2. **可用价值仅剩两条**：
   - obs_metadata（6.07GB，3.02 亿细胞）按 9/30 方案做**上下文指纹/数据集相似度
     特征**——但注意 cell_type 大面积空（本地 10 SRX 仅 telohaec 两样本与
     aics_SRX22915753 有实体注释，hek293ft/mdamb 全空），指纹质量打折；
   - 若未来确需这 5 个靶的扩源，须回到 GEO 原始作者数据取 guide 分配表
     （CPA 筛选与 AICS 研究均有原论文 FASTQ/矩阵），属独立数据工程，不建议排入当前主线。
3. fid 方向一致性问题的解法不变：优先 7 已有源响应基因级置信加权
   （IMM 咨询线）+ 官方 zeroshot 检查点校准，scBaseCount 不入该链条。

## 五、工程留档

- 完整性修复过程与验证细节：`docs/team/cb_scbasecount_完整性更新.md`。
- 下载脚本：`scripts/dl_scbasecount_repair.sh`（串行，含 curl -q 修复）、
  `scripts/dl_obs_metadata_parallel.sh`（24 路并行分片，注意其 seq -f 拼接 bug）。
