# VCC 2026 · Arc 虚拟细胞竞赛参赛仓库

> 第二届 Virtual Cell Challenge（零样本跨细胞类型 CRISPRi 响应预测）
> 关键日期：**10/22** 最终测试集发布 · **11/5 23:59 UTC** 截止 · 每队每天限交 2 次

## 任务一句话

对验证上下文 A/B/C（以及最终 D/E/F）各 300 个敲低靶基因，**仅凭该上下文的 NTC 对照表达谱 + 靶基因列表**，预测扰动后 400 细胞的原始计数谱（`h5ad → .vcc` 提交，360,000 × 18,533）。

- 竞赛官网：<https://virtualcellchallenge.org>
- 方案文档：[docs/Arc虚拟细胞竞赛_第二届竞赛方案.md](docs/Arc虚拟细胞竞赛_第二届竞赛方案.md)（技术路线 L0–L4、评分解读、时间表、分工）
- 背景调研：[docs/Arc虚拟细胞竞赛_第二届调研报告.md](docs/Arc虚拟细胞竞赛_第二届调研报告.md)

## 目录结构

```
aivc2026/
├── docs/                # 方案、调研、会议纪要（docs/meetings/）
├── data/
│   ├── raw/controls/    # 官方验证数据：context_A/B/C.h5ad + gene_names.csv + pert_counts.csv + manifest.json
│   ├── public/          # 公共训练数据（Replogle GWPS / VCC2025 / Tahoe-100M …），按来源分子目录
│   └── interim/         # 中间处理产物（签名库、上下文指纹等），可再生成
├── src/
│   ├── data/            # 数据加载 / 预处理 / 特征构建
│   ├── models/          # 各档位模型（L0 基线 … L3 主力），一个方法一个子目录
│   └── eval/            # cell-eval 封装、LOCO 本地评测、格式校验
├── scripts/             # 一次性或编排脚本（下载、解包、批量评测）
├── configs/             # Hydra/YAML 配置，按 模型名/实验名 组织
├── outputs/
│   ├── submissions/     # 待提交/已提交的 .h5ad 与 .vcc（命名规范见下）
│   ├── metrics/         # 每次 LOCO / 线上评测的指标快照 JSON
│   └── logs/            # 运行日志
├── envs/                # 环境定义与锁文件
└── submission_ledger.csv  # 提交台账（根目录，核心协作文件）
```

## 数据现状

`data/raw/controls/` 为官方验证面板 `vcc2026-val-1`：

| 文件 | 内容 |
|------|------|
| `context_{A,B,C}.h5ad` | 各 18,400 个非靶向对照细胞 × 18,533 基因，raw counts（CSR float32），obs: `target_gene` / `context` / `ntc_id`(46 guides) |
| `gene_names.csv` | 18,533 基因，**提交 .var 必须严格按此顺序** |
| `pert_counts.csv` | 300 个待预测靶基因 |
| `manifest.json` | 面板元信息（300 扰动 × 400 cells/pert 的真值由 Arc 保留） |

## 快速开始

```bash
# 1) 环境（待固化，暂用 conda/python>=3.10；依赖：anndata scanpy pandas polars torch vcc cell-eval2 state）
conda create -n aivc2026 python=3.10 -y && conda activate aivc2026

# 2) 跑通 L0 基线（对照均值 → predictions.h5ad）
python src/models/l0_context_mean/make_predictions.py

# 3) 格式校验 + 打包提交物
vcc prep outputs/submissions/<name>.h5ad -g data/raw/controls/gene_names.csv \
      --perts data/raw/controls/pert_counts.csv -o outputs/submissions/<name>.vcc
```

## 协作规范（要点）

1. **git 为主干记录**：大文件一律不进 git（已 ignore）；代码进 `src/`，实验配置进 `configs/`。
2. **产物命名**：`outputs/submissions/YYYYMMDD_<模型>_<说明>_v<N>.h5ad`，同一内容的指标快照同名放 `outputs/metrics/`。
3. **每次线上提交必须登记台账** `submission_ledger.csv`（commit hash、LOCO 指标、线上六指标、当日额度用途），线上涨而 LOCO 跌 = 刷榜信号，回退。
4. **每人每天 2 发提交分配**：1 发当日最佳候选、1 发消融/校准。
5. 分支约定：`main` 保持可运行；开发用 `feat/<名字>-<内容>` 短分支；基线落盘后打 tag。

完整规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。
