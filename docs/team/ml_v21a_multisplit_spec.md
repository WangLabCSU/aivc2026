# ml v2.1-(a) multi-split 方差基建 spec（送审稿，动工前供 REV 过目）

- 日期：2026-10-14；撰写：ml-expert；状态：**待 rev-reviewer 认可后动工**
- 目的：为 R-1 G3 唯一豁免通道供给方差证据。R-1 原文（rev_R1_发档预注册.md L21-22）：
  「任一 LOCO 轴点估计劣化于 v1 冻结基线（同 split、同口径）→ 不发；预注册故意不设噪声容忍带；
  唯一豁免通道: gate_report 在**送审时**已自带该轴的多重 split 方差，且劣化幅度 < 2× 轴间标准差——
  事后补测方差无效。」
- 结论先行：LOCO 的 6 个测试集由 ctx 唯一决定、不可重分（test 侧无 split 自由度），
  **split 方差只能来自训练侧重采样 + 随机种子**。本 spec 把"多重 split"操作化为训练侧设计。

## 1. 设计（冻结后不再改）

- 测试集：与 loco_frozen.json 完全一致（整 ctx 留出，同 y、同 rmse 定义）——点估计与已入档
  FAIL 直接可比，不换口径。
- 每个 LOCO 轴跑 7 个模型：
  - **3 个 90% 训练子采样 split**（按训练侧 5 ctx 分层等比抽样，split_seed ∈ {101, 202, 303}）
    × **2 个 xgb seed** {0, 7} = 6 个 subsample 模型；
  - **1 个全量训练对照**（全 5 训练 ctx，seed=7）——与已入档 seed=0 全量折叠差出**纯种子噪声**，
    用于把"子采样效应"从"种子效应"里分离。
- 不变量：FEATS/PARAMS/ROUNDS/CLIP 与冻结版逐字相同（f1–f38 特征、pseudohuber、depth7、
  lr0.05、600r、sub0.8、col0.8）；唯一自由度 = split_seed 与 xgb seed。
- 90% 而非 bootstrap：bootstrap 期望重复率 ~36.8% 会扭曲有效样本量；90% 无放回是温和对称扰动，
  且天然回答"训练集组成敏感度"。

## 2. 判据应用规则（先于看到结果预声明）

- 记录每折：`rmse_model`、`rmse_g1`（轴常数）、`delta = rmse_model − rmse_g1`。
- **轴方差定义**：`std_axis = std(rmse_model)`（该轴 6 个 subsample 模型的样本标准差）。
- **豁免判定**：已入档冻结折叠的点估计劣化（`delta_frozen = rmse_model − rmse_g1`，loco_frozen.json）
  `< 2 × std_axis` → 该轴启用豁免资格；否则维持 FAIL。冻结点估计永远是裁决锚，
  新增运行只贡献方差，**不得**用新运行均值替换点估计（禁止事后换锚）。
- 全量 seed=7 对照只作噪声分解报告（seed 噪声 σ_seed ≈ |rmse_full_s7 − rmse_full_s0|），
  不进入豁免计算。
- 三轴劣化靶点预算（预登记，防事后解读漂移）：rpe1 delta_frozen +0.1177、jurkat +0.0794、
  hct116 +0.0828 → 各需 std_axis > 0.0589 / 0.0397 / 0.0414 才触发豁免审查。

## 3. 运行与产出

- 42 折（6 轴 × 7 模型）；每折 600 rounds hist，n_jobs=40；后台运行，卡死 2h 即 cancel --yes。
- 产出：`outputs/xgb_amp/loco_multisplit.json`（逐折记录 + std_axis + 豁免判定）+
  下一份 gate_report 附节；不触碰今日 λ47（bizard-v0.2）发档链。
- 明确不做：特征搜索、超参搜索、test 侧任何重排、对 G1/G2 的重裁。

## 4. 已知局限（如实入档）

- 训练侧子采样方差是**训练组成敏感度**的下界近似，不等价于官方评分的多重实现方差；
  R-1 豁免通道文字只要求"该轴多重 split 方差"，本设计是该条款的可执行最强解释。
- 90% 子采样平均会轻微抬高 rmse_model，若三轴 std 不达标，结论是"豁免通道确凿不可用"，
  而不是继续加大 split 数搜显著性（预注册不许收紧后放松，反向亦然）。
