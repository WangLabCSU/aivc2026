# 上游闭卷镜像：cell-eval2 官方指标规格书

> **镜像目的**：竞赛期间引用的 cell-eval2 官方指标文档只以链接形式存在于
> [2026-09-28_官方规则评分与工具链.md](../2026-09-28_官方规则评分与工具链.md)，
> 一旦上游改版/删除，公式出处即断链。本目录为 2026-11-10 抓取的**逐字节原文镜像**（未做任何改写），
> 供闭卷存档与学生学习使用。

## 上游出处（provenance）

| 项 | 值 |
|---|---|
| 仓库 | <https://github.com/ArcInstitute/cell-eval2>（GPU-accelerated evaluation of perturbation prediction models） |
| 目录 | `docs/vcc2026_metrics/` |
| 抓取日期 | 2026-11-10 |
| 规格书最后提交 | `fb141abf` （2026-11-03T02:36:19Z） |
| 当日 PyPI 最新版 | cell-eval2 **0.22.0**（竞赛期间实际使用过 0.16.0–0.22.x 系列，`--preset vcc2026` 口径未变） |
| 上游许可 | 见仓库 LICENSE（Apache-2.0） |

## 文件清单

| 本仓库文件 | 上游原文 | 大小 | 说明 |
|---|---|---|---|
| `cell-eval2_vcc2026-metrics.md` | `docs/vcc2026_metrics/vcc2026-metrics.md` | 45.9KB / 709 行 | **完整规格书**（source document）：六指标定义、参数值（`configs/vcc2026.yaml`）、推导与算例、提交文件要求 |
| `cell-eval2_vcc2026-metrics-brief.md` | `docs/vcc2026_metrics/vcc2026-metrics-brief.md` | 18.2KB / 372 行 | **精简参考卡**：只留定义+参数+实测值，去掉推导。与全文冲突时以全文为准 |
| `cell-eval2_README.md` | `README.md` | 107 行 | 仓库 README：安装、CLI 用法、`--preset vcc2026`、`[gpudge]` GPU extra |

未镜像：`.tex`/`build.sh`（LaTeX 构建链）与 `.pdf`（由上述 md 生成，内容等同）。
如需 PDF，可由镜像 md 重新构建，或访问上游 html 版。

## 阅读路线（学生向）

1. **先读精简卡** `cell-eval2_vcc2026-metrics-brief.md` 的 §0（Scope and parameters）——
   拿到六指标名字、方向（越大越好/可负）、封顶值。
2. **再对照本队实战解读**：
   - [../2026-09-28_官方规则评分与工具链.md](../2026-09-28_官方规则评分与工具链.md) §3 —— 官方口径速查（含每日额度、.vcc 硬约束）
   - [../bench_reliability_final.md](../bench_reliability_final.md) —— 本地复现的可靠性分层：哪些指标能本地算、哪些只能区间管理、仿射换算系数
3. **深挖单一指标**再回全文规格书对应章节：
   - §1 Perturbation discrimination → `pds_cosine`
   - §2 Expression error → `expr_mse_unbiased_capped_norm`（唯一 0–1 封顶指标）
   - §3 The differential-expression table → Wilcoxon + BH FDR 0.05 的 DE 表如何生成（其余四指标皆源于此）
   - §4 Direction fidelity → `de_wilcoxon_direction_fidelity_raw`（可低至 ≈−1.9）
   - §5 Direction reach → `direction_reach_raw`
   - §6 Significant set agreement → `sig_jaccard`（下限 ≈−0.1）
   - §7 Fold-change accuracy → `lfc_nmae`（罚帽 −6）
   - §0/§7 Submission file requirements → raw 整数计数、单细胞 ≤10^6、密度上限等硬约束

## 免责说明

镜像仅用于本队参赛复盘与内部教学。公式与参数的一切解释权在 Arc Institute 上游仓库；
若上游与本镜像不一致，**以上游为准**并更新镜像（附新的 commit sha）。
