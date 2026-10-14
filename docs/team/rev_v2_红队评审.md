# 2026-10-13 REV 红队评审：Bizard v2.0（XGB-MVP 前夕）

审阅材料：iter018 v2.0 设计稿、iter019 换算层探针、iter017 排行榜对齐、submission_ledger.csv、
scripts/official_alignment.py + docs/research/official_alignment_v1.json、scripts/night_chain_1013.sh。
以下每条 CONCERN/FATAL 均附文件+行级证据链。

## A. C2 XGBoost 幅度头标签构造 — CONCERN（结构性，非否决）

1. **核心前提与 iter019 证据冲突**：C2 假设"幅度是约束项、方向全交 v1"（iter018:92-93）。
   但 iter019 已证伪"放大即修复"：k=60 时 pred |Δ| median 已与 ref 对齐，nmae 反升
   （50.2→55.8），mse 恶化 0.9426→1.0562（iter019:36-39，"放大错位 Δ 摊大误差"）。
   该结论产生于 State 源；**v1 的 Δ̂ 是否同样"落错基因"未测**。BizardNet v1.0 的短板是
   pds 0.285/reach 0.071（ledger 行 24）——可分性不足，不纯是幅度。若 v1 方向也错位，
   g 只会把错位 Δ 放大 3 倍，mse 直接受害。MVP 必须先跑 v1-Δ̂×oracle-g 的逐位对拍
   （同 iter019:47 未测的 per-target oracle-k），确认 v1 方向值得放大，再谈 y 构造。
2. **clip[0.5,3.0] 下界过紧**：iter019 的教训是"落错基因的靶应压向 0"，下界 0.5
   禁止杀死坏预测。上界 3.0 是否够宽无任何经验分布支撑（文档无 g 分布实测）。
   MVP 必须报告训练集隐含 g 分布与 clip 边界饱和率；饱和 >10-15% 即需重参数化。
3. **y 的上下文外推无验证**：y=log A（top-200 |logFC| 均值）定义在训练上下文
   （K562/rpe1），官方评测在 A(H1)/B(HepG2)/C(Jurkat)；iter018:79 已证 H1 离训练分布最远、
   零样本最差。XGB 按特征外推 g 到新上下文，正是最脆弱的轴，设计稿无消融项。
4. pseudohuber on log A 对大幅值目标欠权重，而官方 mse 恰由大幅值目标主导
   （iter017:14"大而准的逐扰动效应"）——损失-指标错配，建议按 A 加权或分位数校验。

## B. α 可识别性 — CONCERN（过参数化成立，但失败模式安全）

1. Δ̂ = α·Δ̂_state + (1−α)·Δ̂_v1 再 ×g·λ_global（iter018:94）：α、λ_global、E[g] 三个
   幅度旋钮只有一个可观测量。g 头若自由训练，E[g]≠1 直接吸收 λ_global；
   设计稿未写 E[g]=1 约束或事后重定心——**必须补，否则 LOCO 学到的 (α,λ) 不可迁移**。
2. iter019:45-46 已判 ST 逐基因 Δ 与 ref 不相关 → α·Δ̂_state 是往 v1 上加无关噪声，
   α 的 LOCO 最优解≈0 且方差大。3 个上下文折（A/B/C）拟合 α+λ(+clip) 是拟合噪声。
   好在 α→0 优雅退化为 v1，失败安全——故 CONCERN 非 FATAL。
3. LOCO 能否约束 α：**不能**。α 的信号 = state 相对 v1 的增量信息；iter019 证明该增量
   在逐基因轴上≈0。LOCO 最多输出"α≈0"，与直接置 α=0 等价——建议 MVP 直接 α=0 起步。

## C. State ST 降级判定 — CONCERN（方向对，证据单薄）

1. k 扫描只在 P_H1 做了（iter019:19-30 仅 summary_P_H1.json；Q/R 无 k 曲线），
   而 P_H1 恰是离训练最远的最差上下文（iter018:79）。Q_HepG2/R_Jurkat 零样本
   nmae 2.11/3.10，离 λ 系 1.0 不远，k 响应曲线可能完全不同。n=1 上下文外推到
   "全局换算层证伪"过强。
2. per-target oracle-k 上限未测（iter019:47-48 自认）——这是 C3 最终去留的关键判据，
   且为廉价诊断（复用 probe_state_rescale.py）。**补做 Q_HepG2 的 k 扫描 + P_H1 的
   per-target oracle-k，一天内可完成，再做最终降级**。
3. 但"不入提交链"的判定本身正确且无需等待（零样本全轴弱于 λ 系，iter019:49-50）。
   降级为特征源可以先行，Δ̂ 直接来源的死刑缓期到 Q/R 探针出结果。

## D. LOCO 硬 gate 的 affine 校准 — FATAL（按现规格不可用）

1. **mse 轴零校准点**：7 个官方点 mse 全为 0（official_alignment.py:41-62），校准 json
   中无 mse 键（grep 证实 0 命中）→ gate 的"affine 预测 official overall"对 mse 无预测能力。
   而 mse 分是 XGB-MVP 的整个目标轴（领先者 0.776，iter017:11；我方恒 0，iter017:12）。
   **gate 结构性地看不见 C2 要动的轴**：XGB 即使真实大幅改善 mse，gate 无感；
   即使 mse 爆炸，gate 也无感。这不是样本量问题，是结构性盲区。
2. 即使忽略 mse：7 点中 3 heur 是同一预测器的 λ4.5/5.5/6.5 阶梯、2 fusion 是 v35 的
   λ4.5/5.5——高度共线，有效样本 ≈4 个独立预测器。global 拟合混合家族，而家族偏移
   差异实测 −0.30 vs −0.43（ledger 行 24-25）→ 残差里编码了家族效应，对新家族
   （xgb）外推无效。resid_std pds 0.047 + 总体门限=历史最优 +0.0886 →
   随机绿/随机红的概率都不低。
3. family 拟合 n=2 已退化：fusion pds a=−9.23、fid a=−24.5（两点精确插值，斜率无意义，
   official_alignment_v1.json family 节）——任何走 family 回退的 predict 都是垃圾进垃圾出。
4. 接口 bug 级风险：predict 用的是 anchor 归一化 from_replicate（official_alignment.py:65-78），
   docstring 却称 local_raw（:8, :107）。XGB-MVP 若喂 raw 指标进 predict，口径错位。
   gate 使用前必须修。
5. 修正建议：gate 降级为 (a) LOCO 六指标不劣于 v1 冻结基线（逐项相对判据），
   (b) 幅度 sanity + vcc 干跑，(c) 首发定性为"预登记校准探针"而非"过 gate 才发"。
   affine-overall≥0.0886 仅作参考分，不作为 fire 条件。

## E. 时序风险 — CONCERN（建议：slot1 发 XGB 探针，slot2 留 λ 系）

1. 事实：今日（UTC 10-13）双发已被 night_chain_1013.sh 消费（slot1=λ55、slot2=λ7.5-deep，
   PID 866629）；XGB 最早 10-14 slot1。问题实为 10-14 的双发分配。
2. 收益侧：λ 家族 mse 结构性恒 0（16 发全 0，ledger），**官方 mse 轴信息只有发一个
   幅度类候选才能获得**——这是任何本地消融都替代不了的探针价值。top 队 24-102 发
   （iter017:17），探针信息量本身是排名的一部分；再等一日本地消融的信息增益 < 一发官方 mse 响应。
3. 风险侧：新架构家族首发历史全部负迁移（bizard v01 −0.0173、CCRN −0.0408、fusion v1 −0.0965），
   XGB-MVP 首发期望分大概率 < +0.0886 现状。但首发目的不是涨分是标定——
   若产生第一个 mse≠0 点，calibration 层获得新政权重的锚点。
4. 建议：10-14 slot1 = XGB-MVP 预登记探针（不设 affine 绿灯门槛，改用 D-5 判据）；
   slot2 = λ 家族备份（λ9-deep 或视 slot1 结果），**不要**双发都投 XGB 变体。

## F. 材料未列风险（按严重度）

1. **bench 过拟合复发风险（FATAL 级警惕）**：CCRN bench +0.04 → 官方 −0.09（ledger 行 25）、
   fusion v2 bench 反转（行 8）——"LOCO 上调参→官方反转"已发生两次。XGB 的 50 特征
   选择若在 LOCO 上迭代挑选，是第三次踩同一坑。对策：特征集在 LOCO 前冻结，
   LOCO 只做验收不做搜索；TPE-BO 只许动 α/λ 两个标量。
2. **100k 步无中途 ckpt 单点故障**（iter018:60-62）：无 val split，训练崩=全损；且
   ML 稿规格（hidden 328/30-50k 步）与实际运行（672/100k 步，iter018:101-104）分叉后，
   ML 稿的算力/预算假设全部失效，v2.0 排期基于旧规格。版本管理需一页对齐表。
3. **×1.4 λ 移位律与 g 头的双重校正**：λ_global 本地调优后 ×1.4 移位（iter018:22），
   但 g 是在本地尺度上学的——若 ×1.4 本质是家族幅度偏差，g 与移位律会叠加过校正；
   ens 家族 official/local 比尚未验证（ledger 行 23 自认"待更多点"）。融合家族首发的
   λ 移位假设未标定。
4. **scbasecount 特征的地基不稳**：obs_metadata.parquet 损坏 + 4 研究 .part 未下
   （iter019:63-75），特征化将基于 6/10 有偏子集；sample_metadata 的 perturbation 字段
   CRISPR 标签仅 ~3k/35k 且脏（iter019:72-73）——标签噪声直接进 XGB 的 y。特征化
   若编码 study 身份，4 研究补齐后分布漂移。
5. **xgb-MVP 与 state 重训的版本耦合**：pert_features_se600m.pt 与 state run_v2 的
   checkpoint 版本强耦合（iter018:103），任何一侧重跑都可能让另一侧特征/预测失效，
   需要显式版本 pin（当前文档无记录）。

## 总裁决：需修正后执行

XGB-MVP 本体（特征+标签+pseudohuber+LOCO）可继续开工，但以下四项修正为发档前置：
1. 重定义 gate（D-5）：废除 affine-overall≥+0.0886 作为 fire 条件，改逐项 LOCO 相对
   判据 + 预登记探针定位；修 official_alignment.py 的 anchor/raw 口径错位。
2. g 头加 E[g]=1 约束（或事后重定心），明确 g 与 λ_global 的可识别性分层（B-1）。
3. C3 拓扑合稿前补两个廉价探针：Q_HepG2 k 扫描 + P_H1 per-target oracle-k（C-2）；
   在出结果前 α 默认 0，融合公式中的 state 项不进 MVP。
4. MVP 交付物必须含：g 分布与 clip 饱和率、v1-Δ̂×oracle-g 逐位对拍（A-1/A-2）。
10-14 slot1 按 E-4 分配：XGB 探针 + λ 家族备份，双发不投同族。
