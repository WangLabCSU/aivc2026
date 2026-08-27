# 第二届 Arc 虚拟细胞竞赛（VCC 2026）参赛方案

> 制定日期：2026-08-27 ｜ 配套文档：《Arc虚拟细胞竞赛_第二届调研报告.md》
> 竞赛已启动 1 周，距最终测试集发布约 8 周，距最终提交截止（11 月 5 日 UTC 23:59）约 10 周。

---

## 一、任务一句话拆解

对 6 个匿名细胞系中的 3 个验证上下文（A/B/C）各 300 个 CRISPRi 敲低靶基因，**仅凭该上下文的未扰动（non-targeting control）表达谱 + 靶基因列表，预测每个敲低扰动后 400 个细胞的原始计数表达谱**。预测以 `.vcc` 格式提交到实时排行榜；10 月 22 日发布另外 3 个最终测试上下文（D/E/F，同样只给对照态 + 靶基因列表），11 月 5 日截止。

- 已随本方案归档的 `controls.zip` 即为官方验证数据：
  - `context_A/B/C.h5ad`：各 18,400 个 NTC 对照细胞 × 18,533 基因，`X` 为 **float32 CSR 存储的原始整数计数**（单细胞总量约 0.8–2.7 万 UMI），`.obs` 三列：`target_gene`（全为 non-targeting）、`context`、`ntc_id`（46 个非靶向 guide）；`.var` 为空。
  - `gene_names.csv`：18,533 个基因名（预测输出必须严格按此顺序排列 .var index）。
  - `pert_counts.csv`：300 个待预测敲低靶基因（gene symbol）。
  - `manifest.json`：panel 元信息——每上下文 300 扰动 × 400 cells/pert 真值由 Arc 保留，NTC 每个 46 个 guide ID。
- 三个 h5ad 结构完全一致（各约 210–226 MB）。

## 二、提交格式与技术红线（.vcc / vcc CLI）

一次提交 = 一个覆盖全部 3 个上下文的 h5ad 打包：

| 项目 | 要求 |
|------|------|
| 规模 | 360,000 细胞 × 18,533 基因（300 扰动 × 400 细胞 × 3 上下文） |
| `.X` | **raw counts 空间**的非负整数值（float32 存储），稀疏存储时省略显式零 |
| 体量上限 | ≤ 4.75B 个存储条目，即平均每细胞 ≤ ~13,200 个存储条目 |
| 每细胞上限 | ≤ 1,000,000 counts |
| `.var` index | 必须精确匹配 `gene_names.csv` 的顺序 |
| 不允许 | 出现 non-targeting 行；缺失 pert_counts.csv 中任何一个扰动；数量 ≠400 |
| `.obs` 列 | `target_gene`（gene symbol 字符串）、`context`（字符串 A/B/C） |

CLI 流程（VCC 官方提供 **`vcc` Python 包**，专为编码智能体设计了 skill 文档，可直接喂给 coding agent）：

```bash
pip install vcc  # 官方命令行工具
vcc prep prediction.h5ad -g gene_names.csv --perts pert_counts.csv -o prediction.vcc
vcc submit prediction.vcc -m "team message" --wait   # 返回排行榜结果
```

- **限额：每队每天 2 次提交**（UTC 午夜刷新），同一时刻只能有 1 个在途任务（用 `--wait` 阻塞等待）。
- 用 `cell-eval2` 可在本地完全复现榜单打分（无真值时用于自检格式 + 与基线对比）。

## 三、评分体系解读（决定所有技术取舍）

新版 `cell-eval2` 的 `vcc2026 profile` 共六个指标，按"表达式准确性 / DE 准确性 / DE 方向性"三组平衡设计，总体分 = 六指标在 3 个上下文上的**无权重平均**。六指标：

| 组 | 指标 | 衡量什么 |
|----|------|----------|
| 表达 | **pds** 扰动区分度 | 在 300 个预测中能否把真实效应按余弦距离排到前列 |
| 表达 | **mse** 表达准确度 | 采样噪声底校正后的平方距离，按真实效应大小归一（封顶于真实实验水平） |
| DE | **nmae** log-FC 误差 | 参考显著基因上 log₂FC 的 MAE（越小越好，其余指标越大越好） |
| 方向 | **fid** 方向保真度 | 预测显著基因中方向正确的比例（乘以产出率校正） |
| 方向 | **reach** 方向覆盖度 | 自上而下最深 cut 处方向一致率仍高于阈值的深度 |
| DE | **jac** 显著性重合 | 预测与参考的 DE 基因集合 Jaccard |

三条最关键的"标尺"设定：

1. **参考标度**：每个指标按"上下文均值预测器 = 0 分、真实重复实验 ≈ 1 分"线性映射。即 1.0 就是数据噪声的理论天花板，**超过 1 表示比重复实验一致性还准**。
2. **朴素基线锚点**：主办方一个 robust 朴素模型约得 0.85。意味着直接照抄对照态只能得"及格分"，冲排名必须稳定超越它。
3. **被敲低基因自身所在行从全部六指标中剔除**（这是已知常见的"作弊"维度，官方已规避）。

### 战略推论

- 六个指标里四个半都围绕差异表达（logFC 方向/显著性）而非绝对表达值 → **优先优化 Δ（扰动相对对照的变化量）的质量**，而不是复刻控制态的表达水平。
- pds 是全 panel 层面的排序指标 → 输出的扰动间**相对差异结构**要拉开，宁可锐化预测也不要把 300 个预测做成温和收缩的平均脸（后者 fid/reach/jac 会崩）。但 mse 又惩罚过度激进 → 需要在"信号强度"上校准。
- 无权重的宽聚合是多目标优化问题 → 和 2025 年 Altos Labs 通才奖路线一样，**不做单一指标定制，追求均衡泛化**。
- 每天 2 次提交限额 → 必须建立**强本地评测**做提交前 gate，把线上名额留给确认过改进的候选。

## 四、技术路线（分级推进）

两届获奖经验已经收敛出明确结论：纯端到端深度学习未解决此问题，**冠军方法都是"预训练基础模型/生成式模型 + 经典统计特征"的混合架构**（BioMap xTrimoSCPerturb 基于 scFoundation 改进；Altos Labs PRiMeFlow 在原始表达空间端到端 flow matching）。我们的路线分五级，逐级叠加、随时可停在最高稳定档位提交：

### L0 · 上下文均值复制（当天可交）
`pred = 该上下文 NTC 对照的均值谱（按 UMI 缩放到目标深度）`。预期得分 ≈ 官方标尺下的 0 分参考点。**价值不在分数，而在第一天打通全链路**（格式、vcc CLI、提交流程、榜单返回解析）。

### L1 · 平均响应签名迁移（经典统计基线，第 1 周）
核心假设：CRISPRi 敲低的转录后果在不同细胞类型间共享一个"共识方向"。用公共 CRISPRi 数据（主要 Replogle 2022 GWPS：K562 全基因组 + RPE1 essential，同为 CRISPRi/Perturb-seq）计算每个靶基因的**伪批量 logFC 签名**，跨数据集聚合（同基因跨细胞系的签名相关性作权重的加权平均），再映射回目标上下文：
```
pred(g, context) = 控制态分布采样 ⊕ λ(context, g) × sign_mapped(logFC_public(g))
```
其中 logFC 先映射到该上下文可表达基因集合上（置零不可表达基因）、λ 按效应大小先验缩放。这一档成本低、可解释、是所有后续模型的必备对照组（它就是把朴素基线的 0.85 往上顶的第一杠杆）。

### L2 · State 零样本推理（官方起点，第 1–2 周）
跑通 Arc 官方模型 State（GitHub `ArcInstitute/state`，checkpoint 有 state-24m / state-xl）：输入控制态 anndata + 靶基因，`state tx inference` 直接产出扰动后表达。官方定位它是 benchmark starter（原话大意："若它能赢就不办竞赛了"），但它是：
- 所有自研模型的强 baseline；
- 其状态转移模块可作为混合架构的一个"专家分支"；
- 注意 checkpoint 为**非商业许可**，参赛使用边界需读一遍 license（竞赛用途一般被允许，保留凭证）。

### L3 · 有监督微调 / 上下文条件化（主力档，第 2–5 周）
任务本质 = 学 `f(控制态上下文, 敲低基因) → 扰动后分布`，且要求对未见上下文外推。两条并行支线择优：

**支线 A（稳妥）· State 微调**：用 Lightning config 在精选训练集上继续训练（见第五节数据策略），重点加入多上下文数据增强 + leave-one-context-out 课程学习。

**支线 B（进攻）· 上下文编码器 + 生成式头**（对标 PRiMeFlow 思路，但加混合组件）：
1. 把目标上下文压缩为一组**上下文特征向量**：NTC 对照的伪批量均值、高变异基因谱、通路活性打分（GSVA/AUCell）、细胞周期与应激 signature、Tahoe-100M 同源细胞系的基线嵌入等；
2. 主干网络融合"上下文向量 × 靶基因嵌入（可用 DNA/蛋白语言模型 Embedding 或 gene2vec）"，用一个轻量 **flow matching / 去噪头**在 log-normalized 空间建模响应分布，另用一个**经典统计分支**直接输出伪批量 logFC；
3. 两分支输出做不确定性加权融合（DE 相关四指标吃统计分支，pds/mse 吃生成式分支）；
4. 输出层变换回 counts 空间：从预测上下文的 NTC 分布采样基底，再施加学到的 logFC 修正（天然满足 raw-counts 整数约束和 100 万上限）。

训练目标的设计要点：**损失函数直接对齐六大指标**——伪批量 logFC 的 MAE/L1 项、方向符号项、批内扰动两两余弦的可分性项。

### L4 · 冲刺打磨（最后一周）
- 指标层面的后处理搜索：Δ 强度的全局缩放系数 λ、DE 锐化温度——用最后几次提交额度在小网格上标定；
- 集成：State + L3 两分支的多模型伪批量级融合；
- 冷冻测试期预案：10 月 22 日拿到 D/E/F 后流水线必须**一键运行**（无人工介入、<24h 内出首版提交）。

## 五、训练数据策略（零样本 ≠ 不许训练）

规则允许用任意公开/自有数据训练，唯一禁令是不得使用 6 个竞赛细胞系自身的扰动数据（我们也没有）。组合拳：

| 数据 | 角色 | 备注 |
|------|------|------|
| Replogle 2022 GWPS（K562 全基因组 ~2.5M 细胞 + RPE1 essential） | **主菜**：同为 CRISPRi + Perturb-seq，域最近 | gwps.wi.mit.edu / pertpy 一行加载 |
| 2025 年 VCC 数据集（H1 hESC，30 万细胞，完整 train/val/test 已开放） | 附加上下文 + 已知答案的自测集（其 held-out 任务与本届同构） | Arc Virtual Cell Atlas 下载 |
| Tahoe-100M（50 癌系 × 1100 药物，~1 亿细胞） | 跨上下文泛化先验（药扰≠基因扰动，用于预训练/辅助目标而非主监督） | HuggingFace tahoebio/Tahoe-100M |
| Nadig 2024 / Norman 2019 / Adamson 等经典扰动集 | 小份补充，扩多样性 | Atlas / pertpy 均可取 |
| Altos primeflow-vcc-datasets（CD4 T 细胞 Perturb-seq） | 原代样免疫上下文样本，弥合"未知组织来源"的鸿沟 | HF altoslabs |
| 本地 controls.zip 3 上下文 | 只当"输入条件"和格式参照，绝不参与任何拟合目标 | — |

预处理统一走 **scBaseCount / Atlas 的标准化处理版本**，避免自己重挖 fastq。上下文指纹尽量覆盖多种组织来源（血液/实体瘤/上皮/干性），提高对 D/E/F 的外推期望。

## 六、本地评测与零样本仿真（比模型更优先建设）

官方只给对照态、不给验证真值，所以**本地仿真评测是本项目最重要的基础设施**，所有方案决策都必须经过它：

1. **Leave-one-context-out（LOCO）基准库**：从第五节训练池中挑 ~6 个组织来源多样的细胞系（如 K562、RPE1、H1 hESC、HepG2、iPSC 衍生系、CD4 T），轮流做"伪 D/E/F"：只用其对照态 + 固定一批靶基因列表（尽量取与 pert_counts.csv 重合或功能相近的基因），跑完整管线 + cell-eval 六指标。
2. ** ceilings 对齐**：用 cell-eval 的 `--ceiling`（半分裂 + Spearman-Brown 校正）测公共数据集的噪声上限，把 LOCO 分数换算到与官方一致的"0=上下文均值、1=重复实验"标度，才能可信地预估线上成绩。
3. **提交预算管理**：每日 2 发的分配规则写死——1 发给当日最佳候选、1 发留给消融/校准实验；每次提交前必须在 LOCO 榜单上不劣于当前线上最优对应版本。
4. 版本台账：每次提交记录（代码 commit、配置、LOCO 六指标、线上六指标）进一张表，两周后它就是过拟合检测器（线上涨而 LOCO 跌 = 正在刷榜，回退）。

## 七、工程与资源

- **算力**：L2 State-XL 推理与 L3 微调需要 GPU（单卡 A100/H800 级即可起步；flow matching 训练建议 2–4 卡）。按实验室 HPC 现状排期，8 月底完成环境固化（Docker/uv lockfile，含 vcc、cell-eval2、state、torch 等锁版本）。
- **仓库布局建议**（本次文件已入库 `/data/resource/wanglab/aivc/aivc2026`）：
  ```
  aivc2026/
  ├── docs/            # 调研报告、本方案、会议纪要
  ├── data/raw/        # controls.zip 及解包产物
  ├── data/public/     # Replogle / Tahoe / VCC2025 …
  ├── src/data/  src/models/  src/eval/
  ├── configs/         # Hydra 配置（对齐 state 生态）
  └── outputs/         # 提交物 + 台账 CSV
  ```
- **基线优先级纪律**：每一级 baseline 都要落盘一份 predictions+h5ad 和指标快照，保证任何时候都有"能提交的最强结果"。

## 八、时间表（周粒度）

| 时段 | 里程碑 |
|------|--------|
| 8/27–8/31（本周） | 解包 controls.zip 入库；vcc CLI 全链路打通并交 L0；建 HPC 环境；注册队伍账号 |
| 9/1–9/14（2 周） | L1 统计基线上线（Replogle 签名库构建）；LOCO 基准库 v1 + ceiling 标定；State 零样本（L2）上榜；每天 2 发探明 naive≈0.85 的实际构成 |
| 9/15–10/12（4 周） | L3 主力开发：支线 A/B 并行→中期评测会选主线；LOCO v2 加入更多宿主上下文；集成初试；目标进入排行榜第一梯队 |
| 10/13–10/21（1 周） | 冻结管线；一键复现演练（dry-run D/E/F 场景：假想上下文替换跑通全流程）；留缓冲修 bug |
| **10/22–10/29** | 测试集发布周：24h 内交首版（L1 兜底 + 当日最强模型）；集中算力推理 D/E/F |
| 10/30–11/4 | 冲刺迭代：后处理标定、集成、消融余额投放；**11/4 前锁定最终提交**（避免 11/5 当天队列拥堵风险） |
| 11/5 23:59 UTC | 截止；整理方法文档备赛 Rego/答辩材料，复盘报告 |

## 九、团队分工建议（按 3–5 人配置）

| 角色 | 职责 |
|------|------|
| 建模（×2） | 支线 A（State 微调）/ 支线 B（生成式+统计混合）分头负责，共享 LOCO 榜 |
| 数据工程（×1） | Replogle/Tahoe/Atlas 下载清洗、签名库、上下文指纹库 |
| 评测与工程（×1） | cell-eval 封装、LOCO 自动化、提交台账、CI 回归 |
| PI/协调 | 目标取舍、算力调度、对外沟通；冲奖则需明确名次 KPI 与放弃线 |

人力紧张时的最小可行配置：1 人数据+评测、1 人建模主线（L1→L3-A）、1 人机动。

## 十、风险清单

1. **许可证风险**：State checkpoint 为非商业许可，参赛合规性提交前书面确认；队规允许用第三方模型时再绑定支线 A。
2. **刷榜陷阱**：验证榜（A/B/C）与最终榜（D/E/F）独立计分且不可比，防止对 A/B/C 过特化——LOCO 分数有一票否决权。
3. **提交限额**：队列堵塞/失败不计次数但要时间；最后一次提交至少提前一天完成。
4. **counts 空间陷阱**：忘记转回原始计数、.var 顺序错位、密度超限导致 `vcc prep` 拒收——这些错误浪费提交日历，CI 里做静态 schema 校验。
5. **上下文信息不足**：如果 D/E/F 组织来源远离训练分布，所有数据驱动方法同步失效——L1 统计签名的兜底意义即在此，永远保留一条不含神经网络的提交线。
6. **时间线**：方法迭代黄金期只有 9 月至 10 月中旬，任何基建拖延直接吃掉冲刺窗口。

## 十一、参考资料

- 官方：virtualcellchallenge.org（首页/evaluation 页）；arcinstitute.org/news/virtual-cell-challenge-2026
- 提交工具：`vcc` CLI（PyPI，附 coding-agent skill 文档）
- 评测：github.com/ArcInstitute/cell-eval（cell-eval2，vcc2026 profile）
- 起点模型：github.com/ArcInstitute/state（state-24m/state-xl，Lightning+Hydra）
- 数据：virtualcellatlas.org；gwps.wi.mit.edu（Replogle 2022）；HF tahoebio/Tahoe-100M；HF altoslabs/primeflow-vcc-datasets
- 获奖 preprint：BioMap xTrimoSCPerturb（2025 冠军）；arXiv:2604.13986 PRiMeFlow（Altos Labs，通才奖）
- 相关论文：Cell 2025 纲领（Virtual Cell Challenge: Toward a Turing test for the virtual cell）；Cell 2026 基准（Benchmarking zero-shot generalization）；Adduri et al. 2025 STATE（bioRxiv 2025.06.26.661135）；PerturBench（arXiv:2408.10609）
- 社区教程：gmdbioinformatics.substack.com《Arc Virtual Cell Challenge》系列（Part 3 含 State 微调 notebook，建议持出一个熟悉细胞系如 HepG2 做本地零样本验证）
