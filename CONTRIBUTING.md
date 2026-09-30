# 协作规范（CONTRIBUTING）

团队 3–5 人共享服务器目录（NFS）+ git。原则：**代码进 git、产物落盘、决策留痕、本地评测一票否决线上提交**。

## 1. 目录所有权约定

| 路径 | 谁写 | 约定 |
|------|------|------|
| `src/models/<method>/` | 该方法负责人独占写入 | 一个方法一个子目录，入口统一 `make_predictions.py --config configs/<method>/<exp>.yaml` |
| `src/data/`、`src/eval/`、`scripts/` | 公共设施，改动走 PR + 告知全员 | 是所有人的依赖，向后兼容优先 |
| `configs/` | 各人自己实验自己加 | 文件名含作者缩写或实验意图，如 `l1_signature/ws-20260901.yaml` |
| `data/public/<dataset>/` | 数据工程负责人写入，他人只读 | 附 `README.md` 说明来源、版本、处理步骤 |
| `outputs/` | 谁生成谁命名，不删别人的 | 只追加，永不覆盖 |

## 2. 产物与命名

- 提交物：`outputs/submissions/YYYYMMDD_<model>_<note>_v<N>.h5ad/.vcc`
- 指标快照：同名 `.json` 放 `outputs/metrics/`（LOCO 六指标 + 线上六指标合并记录）
- 日志放 `outputs/logs/`；中间大文件放 `data/interim/`（可再生成，随时可删）

## 3. 台账制度（核心纪律）

每次向排行榜提交后，当日在根目录 `submission_ledger.csv` 追加一行：

```
date,submitter,pred_file,git_commit,loco_score,online_score_per_metric,quota_use,notes
2026-09-01,wsx,l0_context_mean_v1,abc1234,0.00,"pds=..;mse=..;...",daily-best,首日链路验证
```

判读规则：**线上分涨而 LOCO 分跌 → 正在过拟合验证面板（刷榜），该方向回退**。

## 4. Git 工作流

- `main`：始终可运行、可复现当前最强结果；直接 push 仅限 docs。
- 功能分支：`feat/<姓名缩写>-<内容>`，如 `feat/wsx-l1-signature`；合并前自测通过并跑通格式校验。
- 每次产出"可提交级"基线时打 tag：`baseline/L0`、`baseline/L2-state-zero-shot` …
- commit message 用祈使句中文或英文均可，注明动机而非流水账。

## 5. 本地评测 gate（提交前必过）

任何候选在消耗每日 2 发线上额度之前必须满足：

1. `vcc prep` 干跑通过（schema/规模/var 顺序/per-cell counts 上限全绿）；
2. 在 LOCO 基准上不劣于当前线上最优对应版本；
3. 指标快照已落盘 `outputs/metrics/`。

## 6. 环境

- 统一用 `envs/` 内的定义文件创建环境；升级任何核心包（torch/anndata/vcc/cell-eval2/state）先在该文件改版本 + 全员周知。
- HPC 走 SLURM：脚本开头固定 `#SBATCH` 模板（见 `scripts/sbatch_template.sh`），长任务不占登录节点。

## 7. 会议与文档

- 会议纪要进 `docs/meetings/YYYY-MM-DD_主题.md`，决议项列表化（谁、做什么、何时）。
- 方案变更（技术路线调整、指标取舍）更新 `docs/Arc虚拟细胞竞赛_第二届竞赛方案.md` 并在纪要中引用。

## 8. 校准/消融提交豁免条款（2026-09-30 REV 审计后增补）

每日第 2 发可用于**校准实验**，豁免"LOCO 不劣于线上最优"gate，但必须**预登记**三要素于
`docs/iterations/<date>_<tag>-preregistration.md`：①假设；②预期读数（含数值区间）；
③决策规则（每种读数区间对应的下一步动作）。台账 `quota_use=calibration` 仅对已预登记者有效。
