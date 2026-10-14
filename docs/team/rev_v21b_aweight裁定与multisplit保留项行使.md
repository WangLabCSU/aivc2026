# REV 裁定 — v2.1-(b) wA 止损认可 + 42-fold multi-split 独立复核与保留项行使（2026-10-14）

- 撰写：rev-reviewer；对象：`ml_v21b_aweight_spec.md`（78a3f17）、`ml_v21a_multisplit_result.md`、
  `outputs/xgb_amp/loco_multisplit.json`（42/42 折）、`pi_multisplit_C1C2核查.md`（56db89d）
- 效力：R-1 框架内裁定记录；C3 遵守——不重开 aaf52c4 的 G3 FAIL 终裁；本文件只裁定
  方差证据对未来 gate_report 豁免通道的效力。

## 一、v2.1-(b) wA 变体：认可止步，本纪要即产出

REV 逐条核过 (b) 纪要的证据链，**认可正式止损**：不发档、不推进完整 bench+LOCO 对拍。
依据：

1. 动机已被作者自查推翻（修正 1：诚实 y~pred 斜率 0.9699 推翻 pred~y 0.84 衰减伪象）；
   且作者主动废止了自己先前的泄漏口径读数（修正 2：rpe1 wA-ph 0.7751 泄漏作废，诚实值
   0.8270）——负结果纪要里带自我纠错，可信度加分。
2. 4 臂 holdout：wA 把斜率推到 ~1.00 的同时总 rmse 恶化（0.3433→0.3510）、q1 偏差恶化
   （+0.348→+0.420）——本质是 q1→q5 的零和权重再分配，不是改进。
3. jurkat 翻转（wA-ph s0 −0.0840 PASS）种子脆弱：s7 边际 FAIL（+0.0096）；在 multi-split
   已量化轴级 seed 位移 ±0.07~0.18 的前提下，任何单 seed 翻转证据不构成推进依据。
4. rpe1 诚实口径改善不足（0.8270/0.8255 仍劣化 g1 0.7599 达 +0.067），hct116 两 seed 均恶化。
   **三 FAIL 轴无一被稳健修复**，完整对拍可预判复现 G3 FAIL——止步符合 R-1 成本纪律。

唯一保留候选：wA-ph（非 wA-mse），前提=折内归一 + ≥2 seeds；如未来激活，须先按 R-1
纪律出预注册（判据、seed 清单、口径），不得径自动工。

## 二、multi-split 独立复核（REV 自原始逐折记录重算，非转录他方结论）

**C1（ddof=1）✓**：六轴 std_axis 与 ddof=1 重算逐轴吻合；ddof=0 会低估 ~9%（gwps
0.0902 vs 0.0988 等），json 用的是 ddof=1。

**C2（逐折记录+分解）✓**：42 折记录字段齐备（ctx/kind/split_seed/xgb_seed/n_train/
rmse_model/rmse_g1/delta）；逐折 delta 恒等式（rmse_model − rmse_g1 = delta）在 4 位小数
舍入容差内全部成立；每轴 decomp（splitXXX.{mean, seed_pair_gap}）与重算全部吻合；full
对照恒等式 full(s7) = frozen_rmse + seed_noise 六轴全成立。跨文档一致性：jurkat full s7
0.6373 与 (b) 纪要 frozen s7 同值，Δ=0.14266 ≈ 0.1427 同源。σ_seed/σ_split 分解与 PI
memo（56db89d）逐轴吻合（jurkat 0.1258/0.0310；hct116 0.0733/0.0164 等 6 轴全对上）。

## 三、保留项行使 — split-only 口径改判

预声明保留项（rev_v21a_multisplit_spec_review.md）条件命中：豁免名义合格轴（jurkat、
hct116）的合并 std 由 seed 分量主导，"split 方差"名实不符。REV 独立重算并行使改判权：

| 轴 | σ_split（seed 均值间，ddof=1） | 2×σ_split | σ_seed | |delta_frozen| | 名义 | split-only 改判 |
|---|---|---|---|---|---|---|
| jurkat | 0.0310 | 0.0619 | 0.1258 | +0.0794 | T | **FAIL（翻转）** |
| hct116 | 0.0164 | 0.0328 | 0.0733 | +0.0828 | T | **FAIL（翻转）** |
| rpe1 | 0.0053 | 0.0105 | 0.0021 | +0.1177 | F | FAIL（终局，任何口径不可豁免） |

改判理由（三条，均为已验证事实）：

1. **seed 分量是轴级系统位移，不是交换噪声**。s7−s0 均值移量轴内方向一致：jurkat
   +0.178 / hct116 +0.104 / gwps +0.135（s7 全劣化）、hepg2 −0.066（s7 全优）——seed×轴
   固定效应。把它计入"split 方差"后，2×std 阈值保护的不再是 split 抽样不确定性，
   豁免通道名实不符。
2. **量化主导**：jurkat σ_seed/σ_split ≈ 0.1258/0.0310（≈4:1），hct116 ≈ 0.0733/0.0164
   （≈4.5:1）；split-only 口径下两轴 2× 阈值（0.0619/0.0328）均被 |delta_frozen|
   （0.0794/0.0828）压过。
3. **改判在最宽松读法下仍成立**：std(split 均值) 本身含半个 seed 方差（split 均值各平均
   2 个 seed），是 split-only std 的高估；σ_split 分解则是下界；两口径夹住真值而结论
   同向（PI 夹逼论证，REV 独立复核成立）。

**裁定**：当前冻结模型的豁免通道**三轴全部落空**——rpe1 终局不可豁免；jurkat/hct116
经保留项改判 FAIL。`loco_multisplit.json` 不进任何 gate_report 豁免。G3 FAIL 维持
（aaf52c4 终裁不变，rpe1 单轴即决定性）。

## 四、(a)-v2 seed-ensemble：认可方向，附预注册条件

方向认可：每折对 K≥5 个 xgb seed 的预测取均值后再算 rmse，使 σ_seed 降至 σ_seed/√K、
std 回归 split 方差主导。42 折实测 ~6 分钟，K=5 成本可控。动工前须预注册并锁定：

- **E1** seed 清单（K 与具体 seed 值）先冻结后运行；
- **E2** 集成定义=逐样本预测均值后计算 rmse，并声明评估对象变更（被 gate 的预测器
  从单模型变为 seed-ensemble；未来豁免证据必须绑定实际提交的同一预测器定义）；
- **E3** 预注册预算沿用 rpe1 +0.1177 / jurkat +0.0794 / hct116 +0.0828，不放宽；
- **E4** 结果无论方向均入档，遵守 C1–C4 与本裁定第三节的 split-only 改判口径。

## 五、送审要求（正式升格）

未来任何**单 seed LOCO 读数**送审，必须附带轴级 seed 不确定性声明（本运行实测系统
位移：jurkat +0.178 / hct116 +0.104 / hepg2 −0.066 / gwps +0.135，轴内方向一致）；
缺失该声明的单 seed 读数 REV 不予采信。（PI memo 已采纳，此处升格为 REV 送审要求。）

## 结论

1. v2.1-(b) 止损**认可**，纪要（78a3f17）入档为 R-1 负结果记录。
2. multi-split 基建 C1/C2 复核**通过**；保留项行使，**三轴豁免全部落空**，G3 FAIL 维持。
3. (a)-v2 seed-ensemble 认可动工，须先满足 E1–E4 预注册。
