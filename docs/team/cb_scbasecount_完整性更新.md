# scbasecount 数据完整性更新（CB，2026-10-14）

> 承接 iter019 PI 核实（docs/iterations/2026-10-12_iter019_state换算层探针.md 末节）：
> 4 个 .part 未完成 + obs_metadata.parquet 损坏。本次全部修复完毕。

## 一、修复结果：全部恢复，无遗留缺失

### 1. h5ad 矩阵文件 4/4 修复（h5py 逐一验证通过）

| 文件 | 细胞数 | 基因数 | 验证 |
|------|--------|--------|------|
| hek293ft_SRX24879306.h5ad | 113,914 | 36,601 | h5py OK, X shape/attrs 读通 |
| telohaec_SRX22713290.h5ad | 86,389 | 36,601 | h5py OK |
| telohaec_SRX22713298.h5ad | 92,480 | 36,601 | h5py OK |
| mdamb361_SRX19325980.h5ad | 33,629 | 36,601 | h5py OK, 404,529,870 bytes |

- 源：GCS `arc-institute-virtual-cell-atlas/scbasecount/2026-01-12/h5ad/GeneFull/Homo_sapiens/`（匿名 HTTPS）。
- 现状：`data/public/scbasecount/` 10/10 h5ad 全部完整（10 个研究 × 36,601 基因，总计 771,599 细胞）。
- 陈旧 `.part` 文件已清理。

### 2. obs_metadata.parquet (6.07GB) 重下完毕（pyarrow 验证通过）

- `data/public/scbasecount_metadata/obs_metadata.parquet`：6,074,238,304 字节，
  **302,277,013 行 × 10 列**，footer magic PAR1 完整，289 个 row group。
- 损坏旧件保留为 `obs_metadata.corrupted_479MB.parquet.bak`（可删）。

### 3. 根因与工程教训

- **`~/.curlrc` 含 `--max-time 300`**：单流下载 >5 分钟必被掐断且 resume 计数错乱，
  这就是 9/30 那 4 个 h5ad + obs_metadata 下载失败/截断的根因。修复：`curl -q` 忽略 curlrc。
- **GCS 对单连接限速 ~15KB/s**（本环境实测）：obs_metadata 单流需 ~2 天。
  改用 **24 路并行 range 分片下载**（50MB × 116 片，xargs -P24，逐片尺寸校验+重试，
  断点续传），全程 4.2 小时完成。脚本：`scripts/dl_obs_metadata_parallel.sh`。
- 注：脚本自带的 seq -f 拼接有 bug（%04d 转义），实际用 printf 循环拼接；
  最终以 pyarrow 全文件打开 + 10/10 SRX 细胞数对账作为放行标准。

### 4. obs_metadata 交叉验证（全部通过）

- 与 sample_metadata 一致的总行数口径（3.02 亿细胞）。
- 10 个本地 SRX 细胞数与 obs_metadata 逐一 **10/10 MATCH**。
- barcode 级联动验证（telohaec_SRX22713290）：h5ad obs/_index 86,389 个 barcode
  与 obs_metadata 该 SRX 的 86,389 个 cell_barcode **集合完全一致**。
  → **per-cell barcode→(SRX, cell_type) 联动通路打通**。

## 二、数据可用性增量发现

1. **obs_metadata 无 perturbation 列**（列=cell_barcode, SRX_accession, gene/umi_count×3 层,
   cell_type, cell_ontology_term_id）→ 扰动标签仍只能从 sample_metadata（研究级，脏）
   + 文献核实获得，本次结论与 9/30 评估一致。
2. **cell_type 大面积为空**：10 个本地 SRX 中仅 telohaec 两个样本与 aics_SRX22915753
   有实体注释（smooth muscle cell 84,400/90,765；neuron 67,872 等），其余全空。
   hek293ft 4 样本、mdamb 4 样本全部空 → scBaseCount 的注释质量显著低于预期。
3. 竞赛方（Arc）在 2026-01-12 快照里 obs 层质量未升级，scBaseCount 作为
   **签名扩源**的价值主要取决于 h5ad 矩阵本身的扰动研究设计（见特征化报告）。

## 三、遗留

- 无完整性遗留。feature 化与扰动核实结论见配套报告
  `docs/team/cb_scbasecount_特征化报告.md`。
