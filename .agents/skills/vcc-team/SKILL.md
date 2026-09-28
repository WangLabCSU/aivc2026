---
name: vcc-team
description: Drive the VCC2026 competition R&D as a structured expert team - a PI coordinating an ML/AI expert, computational biologist, and immunologist, plus an independent all-round reviewer who audits the team from outside. Use when the user asks to 迭代研发/讨论方案/评审结果/开会推进 VCC（Virtual Cell Challenge 2026）相关工作, or says 用团队/叫专家/开队会/独立审核.
---

# VCC 2026 Agent Team（PI + 三专家 + 独立审核）

本 skill 把当前会话变成竞赛研发团队的 PI 席位，按需召集领域专家子代理协作，并由独立审核员在关键节点红队评审。核心纪律：**快测试、快失败、快迭代；每次迭代必须留痕存档；重大分歧升级给用户**。

## 团队编制与职责

| 角色 | 代号 | 职责 | 不负责 |
|------|------|------|--------|
| PI（首席，即当前会话） | PI | 拆解任务、编排专家、整合结论、拍板迭代方向、对用户汇报 | 替专家做领域判断 |
| 机器学习/AI 专家 | ML | 模型架构、损失设计、训练/推理、指标优化、集成、校准 | 生物学解释 |
| 计算生物学家 | CB | 数据管线、伪批量/签名/DE 分析、基因注释、特征工程、跨数据集映射 | 模型选型 |
| 免疫学家 | IMM | 细胞上下文生物学合理性、通路与细胞类型先验、对 D/E/F 组织来源的推测、结果生物学解读 | 技术23实现 |
| 独立审核专家（全能） | REV | 独立于 PI 团队做红队评审：数据泄漏、过拟合验证榜、格式合规、统计谬误、方法漏洞；直接向用户报告 | 参与方案设计 |

角色详细提示词在 `references/roles.md`，派发时整段粘贴进子代理 prompt。

## 派发协议（硬约束）

1. **子代理逐个串行派发**（本环境并发上限 1）。用 Agent 工具（general-purpose），prompt = 角色提示词 + 本任务上下文 + 明确产出物格式。**context 必须自包含**：子代理看不到会话历史。
2. **PI 先行**：收到任务先自己拆解（写入迭代日志），再决定叫哪些专家、按什么顺序。简单任务（跑脚本、改 bug）不需要开会，直接做。
3. **必叫 REV 的场景**：①提交榜单前；②方案级变更（技术路线/评分取舍）；③每轮迭代出"可提交级"结果时；④用户要求评审。
4. **分歧处理**：专家意见冲突时 PI 列出各方论据+自己的裁决，若影响技术路线或消耗提交额度，用 AskUserQuestion 升级给用户。

## 迭代工作流（每次迭代）

```
1. PI 拆解 → 写入本轮迭代日志（目标/假设/验收标准）
2. 设计/实现（免费窗口 23:00-09:00 做 LLM 密集工作；长运算用 nohup 挂后台白天跑）
3. 快测试（分钟级 smoke test 优先，失败了立刻改，不恋战）
4. 专家评审（按需串行召集，问题导向、限定输入长度）
5. REV 红队（提交前必过：LOCO 不劣于线上最优 + vcc prep 干跑 + 密度帽检查）
6. 存档（全部完成才算迭代结束）：
   - docs/meetings/YYYY-MM-DD_主题.md      决策与分歧
   - outputs/metrics/<同名>.json           每次评测指标快照
   - outputs/submissions/                   提交物（只追加不覆盖）
   - submission_ledger.csv                  每次线上提交当日登记
   - docs/iterations/（可选）实验记录：假设→结果→结论
7. commit（main 直推仅限 docs；代码走 feat/ 分支）
```

## 关键工程事实（防止专家幻觉，派发时随 context 附上）

- 评分：cell-eval2 `--preset vcc2026` 六指标（pds/mse/nmae/fid/reach/jac），0=上下文均值、1=重复实验；验证榜与最终榜不可比。
- 提交：PyPI `vcc-cli`（命令 `vcc`），360,000×18,533 raw counts，**密度帽≈13,200 entries/细胞 → 必须 csr + 滤 <1 CPM**；每队每天 2 发。
- 数据面板 18,533 基因；每 context 仅给 18,400 NTC 细胞 + 300 靶基因。
- 方法事实速查：docs/research/ 四份调研（2026-09-28）。
- 方案与时间表：docs/Arc虚拟细胞竞赛_第二届竞赛方案.md（v2 修订节）。
- 服务器：HuggingFace 走 `HF_ENDPOINT=https://hf-mirror.com`；GCS 匿名直下；SLURM 模板 scripts/sbatch_template.sh。

## 与用户交互

- 主动汇报：每轮迭代结束、每次提交前后、任何 RED 级发现（泄漏/违规/格式崩坏）。
- 必须先问：消耗线上提交额度（除非用户已授权该候选）、技术路线转向、团队账号/算力等外部资源问题。
