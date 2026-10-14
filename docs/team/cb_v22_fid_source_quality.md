# cb v2.2 — fid 签名源质量置信度加权（IMM）+ 官方 zeroshot 校准数据侧准备 spec

- 日期：2026-10-14；撰写：cb-expert；状态：**spec 待 REV 认可，认可前不动工**
- 上游依据：`docs/research/2026-09-30_大数据大模型利用方案.md` §6 fid 解法优先级链（7 源响应基因级置信权重 + IMM 咨询）、`docs/meetings/2026-09-30_IMM咨询与上下文诊断.md`（IMM 加权设计原文）、`docs/iterations/2026-10-14_iter021_bizard-v0.2官方成绩与affine-OOS验证.md`（官方 fid −0.01817 锚点）
- 边界：不触碰冻结的 BizardNet 方向头与 λ 链（λ47=bizard-v0.2 已发档，官方 λ 曲线与 affine 仪器在档）；模型侧实现归 ml-expert，本文只定接口约定。

## 0. 问题定义

bizard-v0.2 官方分量 **fid −0.01817**（pds 0.29562 / mse 0 / nmae −0.07978 / fid −0.01817 / reach 0.07742 / jac −0.00115）。fid 是六分量中少数仍显著为负、且有明确机制解释的分量：

- IMM 咨询机制：竞赛 val 锚 A/B/C 为 p53 有功能的非造血上下文；K562=TP53 缺陷，p53 轴靶在 K562 签名中近乎不响应。签名库中 K562 系来源（gwps、k562_ess、norman 均 K562 系）对这些靶的方向证据缺失或反向 → 聚合后显著集方向近似随机 → fid 深负。
- 本地实测支撑（`outputs/metrics/ntc_context_diagnostics.csv`）：P53_axis 模块分 A 0.14 / B 0.402 / C 0.387 / Q_HepG2 0.221 vs K562 **0.020**、R_Jurkat 0.023——K562/Jurkat 上 p53 轴确实不响应。次级机制：C 上下文高 IFN（0.757）、B 上下文 EMT/上皮态（0.90/0.997）。

## 1. 现有资产盘点（全部已建成，无新增下载）

| 资产 | 路径 | 内容 |
|---|---|---|
| 7 源签名 parquet | `data/interim/signatures/` | gwps、hct116_all、hepg2、jurkat、k562_ess、norman、rpe1（另有 tahoe_p12 可作扩展源）；约定：per-target raw-count sum → CPM×1e4 → log1p − NTC log1p；min_cells=30 |
| l15_fusion sigstore | `data/interim/l15_fusion/sigstore_{gwps,hepg2,jurkat,k562_ess,rpe1}.npz` | 稀疏 {keys: int64 flat idx, data, target_names, n_vocab}；gwps 9,866 靶/81.2M 条目；hepg2/jurkat/rpe1 各 2,393 靶；k562_ess 2,057 靶 |
| confv2 投票机制模板 | `data/interim/confidence_v2.parquet` | (9913, 8)：gene, n_sig_gwps, n_sig_hct116_all, dir_gwps, dir_hct116_all, n_votes, agree_sum, w（w 下限 0.7=β 软收缩）——per-gene 投票机制模板 |
| NTC 模块诊断 | `outputs/metrics/ntc_context_diagnostics.csv` | 上下文×模块矩阵（P53/IFN/EMT/Epithelial/…），源-上下文相似度 ρ 的原料 |
| 聚合接口 | `src/models/l1_signature/make_predictions.py` `load_delta` | 已有 `sign_fix`、`dataset_weights`（Σw·Δ/Σw）可选 config 键；键缺省时与旧聚合逐位一致（5.27M 行回归已验，iter014 入档） |
| 官方检查点 | `data/public/models/ST-HVG-Replogle/{zeroshot,fewshot}/{hepg2,jurkat,k562,rpe1}` | **zeroshot 四上下文本地已齐**（较 9/30 记录的"仅 zeroshot/hepg2"是更新）；每个含 config.yaml / eval_best.ckpt / batch_onehot_map.pkl / cell_type_onehot_map.pkl / pert_onehot_map.pt / var_dims.pkl |

 roster 澄清：9/30 方案的"7 源"名单含 adamson 不含 hct116；本地实况是**已建成 7 源 = {gwps, hct116_all, hepg2, jurkat, k562_ess, norman, rpe1}**，adamson 未建。本 v2.2 以本地 7 源为准；adamson（K562 系，建了会把 K562 系占比抬到 4/8）仅在 REV 要求时作第 8 源补建。

**已闭环路线（iter014 入档，本 spec 不得重开）**：幅度收缩（confv2 全局）、top-K、方向先验、翻号（sign_fix）；C1 sign-fix −0.034 FAIL（reach 减半——双源对消是受保护的防过冲正则）；C2 fusion@λ5.5 −0.031 FAIL；C3 HCT116 全局臂非对称权重 0.7/1.3 −0.018 FAIL。本设计与闭环路线的区别：**逐基因**（非全局常数臂权重）、**数据驱动**（NTC 相关性，非手调 0.7/1.3）、**作用于定显著集之前**（非事后缩放）、**source-LOPO 门控**（不通过即不集成）。

## 2. 设计 A：IMM 源质量置信度加权

### A.1 数据流

```
7 源签名 parquet ──对齐基因词表/靶空间──> 三维矩阵 Δ[src, target, gene] + FDR[src, target, gene]
        │
        ├─(1) 显著门控: sig = (FDR<0.05) & (|logFC| > τ)，τ 预注册 0.25（与 confv2/dirprior 一致）
        ├─(2) 源质量 ρ[src, ctx]: 每源 NTC/基线表达谱与目标上下文 NTC 模块谱的 Spearman 相关
        │       （ctx ∈ {A,B,C}，模块分从 ntc_context_diagnostics.csv 取；K562 系源在 p53 轴靶上
        │        因 p53 轴不响应而与 A/B/C 的 NTC 反应谱天然低相关 → 自动降权）
        ├─(3) 加权共识: Δ*_{t,g} = Σ_{s∈sig} ρ_{s,ctx(t)}·Δ_{s,t,g} / Σ_{s∈sig} ρ_{s,ctx(t)}
        ├─(4) 逐基因置信: N = |sig 源数|；agree = Σρ(与 sign(Δ*) 同向的 sig 源)/Σρ；
        │       w = 2·agree − 1, clip[0,1]；N<3 → 置中性 + 软收缩 w' = β + (1−β)·w（β=0.7 沿 confv2）
        └─(5) 输出产物（§A.3）
```

关键不变量：**权重作用于定显著集之前的 Δ**——下游显著集/方向集判定消费 Δ*（加权共识），而非任何单源或等权均值。1/N 支持度权重被 IMM 否决（混淆覆盖度与置信度），不采用。

### A.2 source-LOPO 门控（预注册，不通过即停）

- 留出 RPE1、HepG2 两源（p53-WT、非 K562 系——恰是对 A/B/C 最该有方向发言权的源）；
- 用其余 5 源的加权共识预测留出源的显著基因方向，读数 = 显著基因符号恢复率；
- 判据：**加权共识恢复率 ≥ 等权基线 + 2pp，且对两个留出源同时成立**，否则停止（记录负结果入档，与 C1/C2/C3 同格式）。留出源自身只作验证，永不进共识。

### A.3 产物（cb 侧交付物）

| 文件 | schema | 说明 |
|---|---|---|
| `data/interim/fid_srcquality/source_context_weights.parquet` | (source, context, rho) | 每源×上下文质量权重，含计算用的模块谱快照列 |
| `data/interim/fid_srcquality/gene_source_votes.parquet` | (target, gene, source, logfc, fdr, sig, dir) | 原始票，从 7 源 parquet/sigstore 重导出 |
| `data/interim/fid_srcquality/consensus_delta.parquet` | (context, gene, delta_star, n_sig, w) | 加权共识 Δ*（若 LOPO 通过） |
| `data/interim/fid_srcquality/source_lopo_report.json` | 逐留出源恢复率、加权 vs 等权对比、ρ 表 | 门控判据证据链 |
| `docs/iterations/<date>_iter0XX_fid源质量加权.md` | — | 结果纪要（无论 PASS/FAIL 都入档） |

### A.4 风险与缓解

1. **过收缩压制真实上下文信号**：β=0.7 软收缩对 N<3 的基因贴近中性，可能削弱单源强信号 → 集成后须在 Q_HepG2 bench 重查 λ（bizard 家族 λ 曲线在档，λ 重校准属模型侧，经 ml-expert）。
2. **K562 系伪一致**：gwps+k562_ess+norman 三份 K562 系源可能在非 p53 靶上互相印证抬权。ρ 降权缓解但不根除；LOPO 留出 HepG2/RPE1 正是 p53-WT 对照检验。
3. **C1 教训（防过冲）**：逐基因加权可能拆掉双源对消保护 → sign_fix 机制保持关闭（闭环路线不重启）；限制权重动态范围（w 上限 1.0、ρ 归一化），reach/jac 设反向守卫。
4. **norman 语义**：norman 是共扰动对屏，靶命名与 300 靶签名库的兼容性未核；先做重叠审计，重叠 <10% 则剔除出投票池并在纪要说明（不是 FAIL，是池定义修正）。

## 3. 设计 B：官方 zeroshot 检查点校准（数据侧准备）

### B.1 目标

官方强基线（ST-HVG-Replogle，llama backbone，HVG ~2000 空间，残差预测 + 随机采样基底）在**我们的 bench**上跑出读数 = bench 校准锚点。约束：检查点 pert 空间 = essentialome ~2,024，**不含竞赛 300 靶全集**，故只能做 bench 校准，不能直接推理竞赛靶（竞赛级需 state tx train 自训，超出本 spec 范围）。

### B.2 数据侧准备（cb 交付）

1. **靶重叠审计（第一步，决定可行性）**：bench 各上下文靶集 ∩ 检查点 pert_onehot_map 键集，逐上下文给出重叠靶数。若 Q_HepG2 重叠 < 50 靶，如实上报并停（锚点统计效力不足）；
2. **输入格式化**：bench 上下文的 raw counts + target 列整理为检查点推理输入；基因空间按 var_dims/X_hvg 对齐；NTC 对照池按 `basal_rep: sample` + `basal_mapping_strategy: random` 配方备好；
3. **onehot 兼容核查**：batch_onehot_map.pkl / cell_type_onehot_map.pkl 对 bench 批次/细胞类型值的覆盖表；未覆盖值的回退策略（映射到最近键或标记 missing）写明，交 ml-expert 定夺；
4. **校准锚点表**：推理输出 → 本地 bench 六分量 → `outputs/bench/official_zeroshot_anchors.json` + 简报 md。

### B.3 分工与边界

- cb：上述 1–4 全部数据侧工作（输入、对齐、兼容表、锚点表）。
- ml-expert：GPU 推理执行、checkpoint 加载细节、把锚点接进 affine 仪器（当前 affine 仪器 n=3 bizard 家族 refit 残差 ≤0.047，方向类分量 ≤0.01——zeroshot 锚点可作为第 4 族参考点提升外推稳健性）。
- 预注册阈值：B1（重叠 ≥50 靶/上下文才继续）；B2（推理在某上下文完成后，锚点读数与该上下文我们融合链读数同表并排，仅作校准参考，**不改冻结 λ 链**）。

## 4. 模型侧集成接口约定（cb 只定契约，实现归 ml-expert）

沿用 `make_predictions.py` `load_delta` 既有模式（iter014 已立：可选键 + 缺省逐位一致）：

1. **新增可选 config 键 `gene_weights`**（建议名，ml-expert 可改）：`{file: <path to source_context_weights/合并权重表>, sources: [...], mode: "pre_significance"}`。语义：按 (source→gene) 权重作用于聚合前的各源 Δ，再走既有 Σw·Δ/Σw 路径。
2. **缺省逐位一致**：键缺省时输出与现行为逐位一致（同 iter014 的 sign_fix/dataset_weights 验收标准；5.27M 行回归为基准模板）。
3. **备选集成路径**：ml-expert 亦可选择将 consensus_delta.parquet 直接作为融合新臂接入 L1.5——此为模型侧设计自由，契约只锁文件 schema（§A.3 表）。
4. **冻结不动**：BizardNet 方向头、λ 链（λ47=bizard-v0.2 已发档）、affine 仪器系数、gate_report 流程。任何官方提交不因本 spec 触发；提交命名仍遵守 `bizard`+版本号规则。

## 5. 预期影响

### 5.1 本地 bench 分量

- 直接作用对象：fid（本地 bench 仪器）与方向类分量（jac、pds 的方向部分）。机制上受益靶集 = p53 轴且方向证据被 K562 系污染的靶（A/B/C/Q_HepG2 上 p53 轴有响应 0.14–0.40，K562 系源给不出该方向）。
- 量级不预承诺，以 source-LOPO 恢复率为先导指标；集成后本地 bench 六分量全表重跑，与 iter014 四条闭环路线的负结果同表对照（防隐性回退）。

### 5.2 官方分量（fid −0.01817）

- 机制：fid 深负源于显著集方向近似随机（K562 伪一致）；源质量加权修正后，fid 有望从 −0.018 向 0 方向修复，pds/jac 不应受损（方向证据变准的直接效果）。
- 可预测性：affine 仪器已 OOS 验证（fid 残差 ~0），本地 bench fid 改善可映射到官方 fid 预期，提交前可出区间带——但**本 spec 不含任何提交动作**，提交决策走既有 REV/R-1 流程。
- 失败路径同样入档：若 LOPO 不通过，fid 维持 −0.018，该线记负结果闭环（与 scbasecount 同等待遇），不追第三轮。

### 5.3 设计 B 的间接收益

zeroshot 锚点入 affine 仪器 → 官方六分量提交前预测的参考点从 bizard 家族 n=3 扩为含官方基线参照，降低下次提交的方向类分量预测残差。

## 6. 工作量与顺序

1. cb：norman/靶重叠审计 + ρ 计算 + gene_source_votes 导出（~0.5 天）；
2. cb：source-LOPO 评测（~0.5 天，纯本地 CPU）→ PASS/FAIL 分叉：FAIL 则纪要入档收线，PASS 则交 consensus_delta + 权重表给 ml-expert；
3. 并行：cb 出设计 B 的重叠审计与输入格式包（~0.5 天）；ml-expert 排期推理；
4. 模型侧集成与 bench 回归：ml-expert（估 1 天，含 no-key 逐位回归）。

## 7. 请 REV 裁定项

1. 设计 A 的加权公式与门控判据（A.1/A.2）是否认可，特别是 β=0.7、τ=0.25、LOPO ≥2pp 阈值三处预注册参数；
2. source-LOPO FAIL 时本线是否即收（cb 建议：收，不追加第三轮）；
3. 设计 B 的执行排期与 ml-expert 的 GPU 时段分配；
4. adamson 是否补建为第 8 源（cb 默认不建）。
