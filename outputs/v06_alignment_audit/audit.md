# bizard-v0.6 State checkpoints 编码器初始化对齐审计（只读审计）

- 日期：2026-10-15；撰写：ml-audit-v06（team-lead 委托，接手前任 ml-expert 紧范围重派）；性质：只读审计，未改代码/配置/冻结资产，未 commit，未提交
- 服务对象：`docs/team/rev_prereg_bizard_v06_dl_delta.md` §4（State checkpoints 作编码器初始化候选）+ `docs/team/impl_v06_dl_scaffold_notes.md` §3（双重错配风险的量化）
- 数据口径：竞赛 panel = `data/raw/controls/gene_names.csv`（18,533 基因，manifest `vcc2026-val-1` n_genes=18533 核对一致）；300 靶点 = `data/raw/controls/pert_counts.csv`（301 行含表头 = 300 靶）

## A. 逐上下文基因空间交集率表

重要口径更正：`var_dims.pkl` 顶层是 dict（非裸列表），有效键为 `gene_names`（= HVG/输出空间基因表）、`pert_names`（= 扰动 onehot 表）、`input_dim=2000`、`pert_dim=2024`。前任结论「~2000 HVG」正确，但审计须以 `gene_names` 键为准。

ST-HVG-Replogle 家族 8 个 best.ckpt 候选：四上下文（hepg2/jurkat/k562/rpe1）的 HVG 表**完全相同**（zeroshot 与 fewshot 同表逐位一致），故交集率与上下文无关，实为一张全局表：

| ckpt/上下文 | HVG 数 | panel 数 | 交集数 | 交集率(ckpt) | panel 覆盖率 | 300 靶在 pert_names |
|---|---|---|---|---|---|---|
| fewshot/{hepg2,jurkat,k562,rpe1} 任一 | 2000 | 18533 | 1877 | 93.85% | 10.13% | 0/300 |
| zeroshot/{hepg2,jurkat,k562,rpe1} 任一 | 2000 | 18533 | 1877 | 93.85% | 10.13% | 0/300 |
| local_retrain/state_se600m_gwps_rpe1（辅助候选） | 8192 | 18533 | 7786 | 95.04% | 42.02% | 300/300 |

补充事实：
- 8 候选 HVG 并集 8256 基因，其中 7839 个在竞赛 panel 内——**编码器输入侧的预训练知识最多只覆盖 panel 的 42%（本地 retrain 候选）**；
- 扰动空间：官方 8 候选 pert_onehot 表 2024 项，**300 竞赛靶 0 命中**；仅本地 retrain 候选（gwps+rpe1，9898 项）300 靶全命中。扰动分支维度（2024 vs 9898）不匹配，重建 onehot 表后 pert 嵌入权重不可直接继承。
- 四上下文「无差异」本身即结构证据：ST ckpt 是 per-context 训练但共享同一 HVG 表（Replogle 全基因组扰动图谱 HVG），错配是**家族级**而非上下文级。

## B. 低交集上下文的处理选项与推荐

背景：这里「低交集」实指 panel 覆盖率（10.1%）而非 ckpt 内交集率（93.9%）——ckpt 的 2000 基因 93.9% 都在 panel 里，问题反向：panel 的 90% 基因不在 ckpt 输入空间内。

- **Option a（映射声明）**：编码器输入按 ckpt 原生 2000 HVG 排列；panel 基因经固定映射表投影：在 ckpt HVG 集内的取真实表达，集外的置零/均值填充，映射表落盘 `gene_align/` 并写进声明。encoder 主干权重可完整继承；集外基因信息经由下游 decoder/输出头（输出空间重训为全 panel）与 6 鸟特征间接补足。
- **Option b（放弃该 ckpt，新鲜初始化评测）**：对交集率不达标的候选直接不用，该上下文 DL 头走随机初始化对照。

**推荐：Option a（全上下文统一映射声明），理由一句**——四上下文 ckpt HVG 表逐位相同且交集率 93.9%，不存在「某上下文特别坏需要单独放弃」的情形，零填充映射的成本（一张对齐表）远小于放弃 8 个已实测可加载的官方 ckpt 的机会成本；同时预注册 §5 门槛 4 的三臂消融天然把「新鲜初始化」保留为 DL-only 臂内的对照，Option b 的科学价值已被消融结构吸收，无需单独执行。

## C. k562_ess 六上下文口径选择

- **Option a（映射声明进口径）**：k562_ess 作为第六上下文直接进 v0.6 评测口径，其 DL 成员初始化复用 k562 官方 ckpt + 同一张映射声明（k562_ess 与 k562 同细胞系，HVG 表适用性最强）。
- **Option b（重新评测定口径）**：对 k562_ess 先单独跑一轮初始化来源评测再定。

**选择：Option a，理由一句**——官方 k562 ckpt 与 k562_ess 是同细胞系最近亲（其余 5 上下文均无更近候选），映射声明与 A 节同表复用零新增工作量；Option b 的评测本身需要先有一个能跑的 DL 头，属鸡生蛋，且会拖慢 3-5 天预算。

## D. 首个 epoch 计划与排期影响

- 模型规模（官方 recipe 原样）：State hidden_dim=328、llama bidirectional backbone 8L/12 heads/intermediate 3072、encoder 1 层 + decoder 1 层、cell_set_len=64、blur=0.05——参数量 ~30-50M 量级（450MB ckpt 含优化器与 onehot 表），**单卡 L20 46GB 远未触顶**，batch 可按 cell_set 64 × 梯度累积放大。
- 首 epoch 预算：训练对主表 = 6 上下文签名级对（~300 靶 × panel 内基因），细胞级 energy loss 支路每上下文 NTC 池随机采样；单 epoch 预计 **1-3 小时**（含 HVG 投影与 basal 重抽），3-5 天预算足够跑完择优 + 三臂消融。
- 门槛风险传导：若 B/C 结论为坏消息（此处结论是好消息），唯一硬影响是 fid BCE 标签构造须只用竞赛上下文数据（scaffold §1.4 已预声明）。
- **排期判断：不受影响**。交集率 93.9% + 映射声明路线成立，编码器初始化候选池 8 个全部可用，无任何候选需要废弃；v0.6 训练按预注册 §8 的 3-5 天正常开跑，10-16 提交窗口照旧以 v0.5 artifact 保底（预注册 §7 回退路径本来就不依赖 v0.6 赶窗口），审计结论不改变该排期。
