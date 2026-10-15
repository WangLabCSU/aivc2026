# cb v2.2 C1 冻结文档 — ρ 定义 / 模块谱 / 权重公式预注册（LOPO 执行前冻结）

- 日期：2026-10-14；撰写：cb-expert；状态：**C1 冻结落盘，按 PI 放行顺序直接进 ②③④⑤，不等二次确认**
- 冻结依据：REV 裁定 C1（`docs/team/rev_v22_fid_source_quality_spec_review.md`）——ρ 定义与
  source_context_weights.parquet 须在 LOPO 执行前冻结，冻结后不得因任何读数调整 ρ/τ/β；
  PI 补充裁定（ρ clip 认可 / 细化一 N 计数 / 细化二 fallback-w / 6 模块逐表溯源为冻结完整性一部分）全部并入本文。
- 上游 spec：`docs/team/cb_v22_fid_source_quality.md`（A.1/A.2 预注册判据，REV 裁定 ① 已认可）。

## 1. 冻结资产清单（含 sha256）

| 文件 | 内容 | sha256 |
|---|---|---|
| `data/interim/fid_srcquality/module_spectra_z_frozen.csv` | 10 模块 × 9 上下文模块 z 谱快照（C1 要求的谱快照母本） | `4274f7576007ac73de1aee20ee4c477063014b5f392bc4af24cefaed4ac3cb8e` |
| `data/interim/fid_srcquality/rho_9ctx_frozen.csv` | 9×9 上下文×上下文 ρ 矩阵（10 点 Spearman） | `b3b25925f00aa0f884346db53be9e010c8d77638f140de8ae9220b4e4cf6cdb2` |

行/列顺序（两 csv 一致）：`[a, b, c, hepg2, jurkat, rpe1, hct116, gwps, k562_ess]`；
模块行序：`[P53_axis, IFN, E2F_Prolif, MYC, EMT_mes, Epithelial, Myeloid, Erythroid, Tcell, Endothelial]`。

## 2. 模块基因列表与逐表溯源（PI 补充要求：逐表注明来源，属 C1 冻结完整性）

### 2.1 v0 四模块（来源：`configs/l15_fusion/v0.yaml` module_sets，逐字取用）

| 模块 | 列表数 | in-vocab | 越表基因 |
|---|---|---|---|
| E2F_Prolif | 25 | 25 | 无 |
| P53_axis | 17 | 17 | 无 |
| IFN | 22 | 15 | IFIT2, IFI44, IFI44L, OAS1, OAS2, MX1, CXCL10（7 个不在 v3 词表） |
| MYC | 10 | 10 | 无 |

### 2.2 新增六模块（来源：原构建脚本逐字恢复，见 §3.3 溯源；基因名单源自 IMM 咨询文档）

| 模块 | 列表数 | in-vocab | 越外词表基因 |
|---|---|---|---|
| EMT_mes | 13 | 13 | 无 |
| Epithelial | 11 | 11 | 无 |
| Myeloid | 8 | **6** | **CSF1R、MPO**（两者均不在 v3 词表；此前记录"仅 CSF1R 越外"有误，以本冻结为准） |
| Erythroid | 7 | 7 | 无 |
| Tcell | 10 | 10 | 无 |
| Endothelial | 4 | 3 | TEK |

合计：列表 127 基因，in-vocab 117。谱值用 in-vocab 子集均值（无缺失值填充）。

十模块基因全表（冻结，逐字）：

```text
P53_axis   : CDKN1A BBC3 MDM2 TP53 GADD45A GADD45B DDB2 SESN1 SESN2 RRM2B PMAIP1 BAX PERP ZMAT3 TRIAP1 PLK3 EI24
IFN        : ISG15 IFIT1 IFIT2 IFIT3 IFI6 IFI44 IFI44L IFITM1 IFITM3 OAS1 OAS2 OASL MX1 MX2 IRF7 IRF1 RSAD2 DDX58 IFIH1 CXCL10 STAT1 STAT2
E2F_Prolif : MKI67 TOP2A PCNA MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 E2F1 E2F2 AURKA BIRC5 CCNB1 CCNE1 TYMS FEN1 RRM2 UBE2C CDC20 CENPF HMGB2 NUSAP1 PRC1 ASPM
MYC        : MYC NPM1 ODC1 CAD LDHA PDK1 SLC1A5 GLS CDK4 TERT
EMT_mes    : VIM FN1 CDH2 SNAI1 SNAI2 TWIST1 ZEB1 ZEB2 ITGA5 ITGB1 SPARC SERPINE1 COL1A1
Epithelial : EPCAM KRT8 KRT18 KRT19 KRT7 CDH1 CLDN3 CLDN4 CLDN7 TJP1 DSP
Myeloid    : CD68 CSF1R TYROBP CST3 CTSS MPO LST1 HCST
Erythroid  : ALAS2 EPOR FECH GYPA KLF1 HBA1 TAL1
Tcell      : CD3D CD3E CD3G TRAC CD2 CD5 CD7 CD28 LCK IL7R
Endothelial: PECAM1 FLT1 ICAM2 TEK
```

## 3. 模块谱计算管线（冻结配方，bit-exact 复现已验）

### 3.1 精确步骤

1. 词表：`data/interim/l15_fusion/v3/vocab.json`（14,372 基因），`v2i = {g: i}`；
2. 上下文：`ctxs = ['a','b','c','hepg2','jurkat','rpe1','hct116','gwps','k562_ess']`；
3. `M = np.stack([basal[s] for s in ctxs])` → **(9, 14372) 上下文主轴**，float32，源 = `data/interim/l15_fusion/v3/basal_{s}.npy`；
4. 每基因跨 9 上下文标准化：`mu = M.mean(0)`，`sd = M.std(0)`（**ddof=0**）；
5. `sd_safe = np.where(sd < 1e-8, 1.0, sd)`（全零/近常数基因分支）；
6. `Z = (M − mu) / sd_safe`；
7. 每模块：`idx = [v2i[g] for g in genes if g in v2i]`；`vals = Z[:, idx].mean(axis=1)` → 9 个每上下文模块值；
8. 列序 `[a,b,c,hepg2,jurkat,rpe1,hct116,gwps,k562_ess]`，写 `module_spectra_z_frozen.csv`；
9. ρ：`scipy.stats.spearmanr` 于两上下文的 10 维模块 z 谱之间（n=10）。

### 3.2 复现性声明（三级）

- **逐位复现**：按 §3.1 配方复跑，10/10 模块与冻结 csv **逐位相等（maxdiff = 0.000e+00，np.array_equal 全真）**；
- **ρ 复算**：由冻结 csv 经 scipy.spearmanr 重算与冻结 ρ 矩阵最大差 **1.11e-16**（1 ULP，非逐位但机器精度内）；
- **解析签名**：n=10 Spearman 分母 n(n²−1)=990，ρ = 1 − 6Σd²/990 = k/165 的整数倍；冻结矩阵全体素（含对角 0.9999999999999999）符合该签名（float 累积最大偏差 2.84e-14）。任何 ρ 值均可由冻结 csv 独立验证。

### 3.3 生成脚本溯源（披露）

生成脚本未入库（建构发生在 subagent 会话的 /tmp 草稿脚本）。溯源闭环：
原脚本从 subagent transcript（agent-acb-expert-0307df3d731fcb25.jsonl line 1003）**逐字恢复**，
恢复版复跑与冻结 csv **全部 10 模块逐位一致（maxdiff 0.000e+00）**——本冻结文档 §3.1 即该配方。
入库复现脚本：`scripts/repro_fid_srcquality_frozen.py`（对两冻结 csv 重算并断言逐位一致），
**已于 2026-10-14 落盘并跑通 PASS**：10/10 模块 bit-exact（maxdiff 0.000e+00）、ρ 重算 maxdiff
= 1.11e-16 = 2⁻⁵³（1 ULP，容差以 `2**-53` 精确值断言）。

## 4. ρ 定义（冻结）

- **ρ[src_ctx, tgt_ctx] = Spearman(模块 z 谱[src_ctx], 模块 z 谱[tgt_ctx])**，n=10 模块点；
- 每源映射到其 basal 上下文：gwps→gwps、hct116_all→hct116、jurkat_all→jurkat、k562_ess_all→k562_ess、rpe1_all→rpe1、hepg2→hepg2；
- **norman 不指派 ρ**：无专属 basal 上下文，且已按 C2 剔出投票池（见 §9）；不为其杜撰 basal 归属。

冻结矩阵关键结构读数（LOPO 前在档，机制预期）：

| 对 | ρ | 机制含义 |
|---|---|---|
| gwps ↔ k562_ess | **0.9515**（157/165） | K562 系伪一致——两源模块谱近共线，互相印证不可作独立证据 |
| rpe1↔hct116 | 0.5515 | p53-WT 非造血对中等一致 |
| hepg2↔hct116 | 0.6848 | 同上 |
| rpe1↔jurkat | −0.8545 | 反向对齐 → clip 后 0 |
| hepg2↔jurkat | −0.7818 | 同上 |
| a↔jurkat | 0.9030 | Jurkat 与 A 锚在模块谱空间反而高相关（Tcell/Endothelial 主导）——说明 ρ 度量的是全模块谱形而非 p53 功能本身 |

## 5. ρ_used 规则（冻结，PI 认可）

1. **clip**：ρ_used = max(ρ_raw, 0)；**永不翻号**（负 ρ 源在该上下文弃权，不给反向权重）；
2. **全 0 回退**：单元内所有 sig 源 ρ_used = 0 → Δ* = 该单元 sig 源等权算术均值（回退 = ρ 退化常数，不另立特殊分支）；
3. **fallback-w（细化二，冻结）**：回退单元 Δ* = 等权均值，**w 用同一 A.1 公式在等权下计算**（agree = 同向 sig 源数/N），N<3 软收缩照旧；
4. **拒绝方案**：(ρ+1)/2 线性重标（已否决入档）、1/N 支持度权重（IMM 否决：混淆覆盖度与置信度）、翻号（闭环路线）。

## 6. 显著门控与权重公式（冻结）

- **显著门控 = |logFC| > τ，τ = 0.25**（confv2 同口径）。**偏差披露**：签名源 parquet 无 FDR 列，FDR<0.05 分支数据面不可计算——PI 认可 fallback："数据面不可计算是事实，非口径选择"，双档（spec 偏离 + PI 认可）；
- **w 公式（A.1 预注册，冻结不动）**：agree = 同向 sig 源 ρ_used 和 / 全体 sig 源 ρ_used 和；w = 2·agree − 1 clip[0,1]；N<3 → 软收缩 w′ = β + (1−β)·w，**β = 0.7 冻结**（confv2 实测 N=1 单元 w=0.7 恰为 β 下限，经验支持预注册值）；
- **细化一（N 计数，锁定）**：N = |sig 源数| 按 A.1 原文 = **clip 前 sig 源数（与 ρ 无关）**。ρ_used=0 的源仍计入 N、贡献 0 权重。**后果明写**：单正 ρ 源 + N≥3 的单元**不触发软收缩，Δ* 实质单源**——确认此为预期行为（门控要检验的正是伪一致 vs 真独立的分辨，单源单元如实呈现其单源性）；
- **细化二（fallback 单元 w）**：全 0 回退单元 Δ* = 等权均值；agree 按等权口径 = 同向 sig 源数/N；N<3 软收缩照旧。回退 = ρ 退化常数，非特殊分支；
- **confv2 历史口径澄清**："confv2 同口径"仅指 τ=0.25 显著门控，**不含 w 公式**——confv2 历史值（N=2 全反对 w=0.5）与 A.1 预注册公式（该情形 w′=0.7）不重合（差 0.2），A.1 公式按 C1 冻结不动，confv2 观察值仅作历史语境记录。

## 7. 被否决的替代方案（防复开）

| 方案 | 裁定 |
|---|---|
| (ρ+1)/2 线性重标（保负 ρ 信息） | 否决（PI 已裁）：负相关源不应贡献正向票权 |
| 1/N 支持度权重 | 否决（spec §A.1 原文 + IMM）：混淆覆盖度与置信度 |
| 翻号 / sign_fix | 闭环路线（iter014），冻结不重启 |
| 全局臂权重（C3 路线） | 闭环 FAIL（−0.018），不得重开 |

## 8. source-LOPO 读数定义（冻结，预注册）

- **投票池（C2 修正后）**：4 源 = {gwps, hct116_all, jurkat_all, k562_ess_all}。spec A.2 "其余 5 源"为笔误，池定义修正非 FAIL（norman ∩ 竞赛 300 = 3 靶 = 1.0% < 10%，∩ essentialome = 0 → 剔除）；
- 留出源：**RPE1、HepG2**（p53-WT、非 K562 系）；留出源只作验证，永不进共识；
- eval 单元 = (target, gene)：留出源 |logFC| > 0.25；投票源同门控（|logFC| > 0.25，无 FDR）；**≥1 票单元**才计入；两法（加权 vs 等权）同分母；
- 读数 = **符号恢复率**（consensus Δ 与留出源 logFC 同号比例）；
- **PASS 判据（预注册）**：加权 ≥ 等权 + 2pp，且对两留出源**同时**成立；任一不成立即 FAIL；
- **morphology E 条款（预声明）**：clip 后两留出源的唯正 ρ 投票源均为 hct116（rpe1↔hct116 0.5515 / hepg2↔hct116 0.6848；gwps/k562_ess 对两留出源 ρ_raw ≤ 0 → ρ_used=0）→ LOPO 实质 = "hct116 单源加权 vs 4 源等权"。**这正是门控要检验的对象，预声明即可，不构成偏差**；
- **非门控诊断列（义务）**：source_lopo_report.json 增逐源单源恢复率列（hct116/gwps/k562_ess/jurkat 各自单独 vs 等权基线），量化"伪一致到底值多少"——纯诊断，不改判据。

## 9. 空间边界披露（PI 裁定 ⑤，预注册不改）

- LOPO 在 essentialome 空间执行（4 源投票池）：{2 票：38, 3 票：623, 4 票：1732}；
- 竞赛 300 靶的投票池仅 {gwps, hct116_all}（255 靶 2 票 / 44 靶 1 票 / 1 靶 0 票，全体 N<3 → 全域软收缩生效）；
- 这是 LOPO 证据向竞赛空间外推的**结构性边界**，如实披露；不改预注册判据、不改池定义、不做事后补池（REV C2）。

## 10. 披露义务清单（LOPO 执行时须量化，不得只字未提）

1. **fallback 单元占比**：Σρ_used=0 的单元数 / 总单元数——K562 系伪一致残余风险的直接读数；
2. N 分布全表（N=1/2/3/4 单元数）与软收缩触发面（N<3 占比）；
3. 逐源单源恢复率（§8 诊断列）；
4. 每单元 sig 源构成中 ρ_used=0 源占比（细化一后果的直接读数）。

## 11. 冻结确认清单（LOPO 前逐项落定）

- [x] ρ 定义 + k/165 解析证明 + 复算 ≤1.11e-16
- [x] 10 模块基因列表逐表溯源（v0.yaml 4 + 原脚本恢复 6）+ in-vocab 计数
- [x] 生成脚本溯源披露 + bit-exact 复现（10/10, maxdiff 0.0）；复现脚本已入库并跑通 PASS（§3.3）
- [x] ρ_used clip / 不翻号 / 全 0 回退 / fallback-w（细化二）
- [x] τ=0.25（FDR 缺失偏差双档）、β=0.7（confv2 N=1 实证）、A.1 公式冻结
- [x] 细化一 N 计数（clip 前计 sig 数；ρ_used=0 计入 N 贡献 0 权；单正 ρ+N≥3 不收缩 = 预期行为）
- [x] LOPO 判据（≥2pp 双留出源 AND 门）+ 池定义（4 源，C2 修正）
- [x] 空间边界（essentialome vs comp-300）披露义务
- [x] 冻结 csv sha256（§1）
