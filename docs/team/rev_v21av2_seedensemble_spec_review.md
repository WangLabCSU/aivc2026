# REV 裁定 — v2.1-(a)-v2 seed-ensemble 评测预注册 spec（认可动工，附 C1–C4）

- 日期：2026-10-14；裁定人：REV；对象：`docs/team/ml_v21av2_seedensemble_spec.md`
  （commit 56ae757，ml-expert 撰写）；裁定送达 ml-expert 后入档。
- 裁定：**认可按预注册动工**（240 训练 ≈35 分钟，CPU），附 C1–C4 条款。
  送审方请求正式裁定的备案点（ensemble 用自身 delta_ens 判定、delta_frozen 降为对照）
  **予以认可**，确认为 E2/E3 联合忠实读法。

## 一、备案点裁定（E2/E3 联合读法）

E2 已声明被 gate 预测器自单模型变更为 seed-ensemble（SET_A, K=5, 逐样本 log 空间算术
均值）。豁免判据必须评估"将来实际提交的预测器"对 G1 的劣化量（delta_ens）；若沿用单模型
delta_frozen 配 ensemble 方差，恰好重建 (a) 暴露、REV 保留项（7909e94）击破的
"名实不符"混合口径（单模型点估计 × ensemble 方差）。spec §2 判定结构与 E2 声明逐字一致，
无矛盾。delta_frozen（0.1177/0.0794/0.0828）仅作冻结单模型对照值继续入档，不进判定。

## 二、脚本预检（scripts/xgb_seed_ensemble.py，未跟踪版，与 spec 逐项核对）

1. E1 清单逐字冻结：SET_A=[0,7,13,42,2026]、SET_B=[1,2,3,4,5]、SPLIT_SEEDS=
   [101,202,303,404,505,606]，与 spec §1 表一致；运行中不加 K/不换 seed/不加 split。
2. keep 索引在成员训练前一次生成（`default_rng(split_seed)` 按 5 训练 ctx 分层等比抽
   90%，行级 (target,ctx) 抽样，同 (a) 协议/REV bba2a7a 焦点 (i)），5 成员共享同一
   keep——"折内差异纯为 seed 分量"的协议在代码层成立。
3. E2 均值定义落地：`np.mean(preds, axis=0)` 逐样本 log 空间算术均值后算 rmse；
   delta_ens = rmse_ens − rmse_g1，g1 取自冻结锚 folds。
4. 逐折落盘字段齐备：split_seed / member_seeds / member_rmses(5) / member_mean /
   rmse_ens / rmse_g1 / delta_ens（Arm B 另含 seed_set），增量写盘可续跑——满足 REV
   独立重算所需的全部字段。无阻疑点。

## 三、C 条款（随认可生效）

- **C1 产物入档**：`outputs/xgb_amp/loco_seed_ensemble.json` git add -f 入档（Arm A
  逐折 5 成员 rmse + Arm B 逐折全量）；结果纪要附判定重算表（逐 FAIL 轴
  |delta_ens(Arm B, SET_A)| vs 2×σ_split_ens），供 REV 从逐折记录独立重算 σ（ddof=1）
  与逐轴判定（C1/C2 同款纪律）。
- **C2 跨臂口径条款**：σ_split_ens 只从 Arm A 六个 ensemble rmse 取（ddof=1）。
  Arm B delta_ens（全量）与 Arm A σ（90% 子采样折）存在跨臂口径差，按 (a) 案 C4 同款
  处理：子采样位移不入判定，仅 σ 进门槛；SET_A−SET_B 位移（n=2）只作不确定性声明，
  不作方向证据（spec §1 已声明，本条款升格锁死）。
- **C3 E2 逐字绑定**：未来任何引用本豁免证据的 gate_report，提交预测器必须逐字 =
  seed-ensemble(SET_A, K=5, 逐样本 log 空间均值, 同 FEATS/PARAMS/ROUNDS/CLIP)；
  单模型读数不得引用本方差。结果无论方向全部入档，判定权在 REV。
- **C4 边界确认**：不动 loco_frozen.json / loco_multisplit.json 锚；不重开 aaf52c4
  的 G3 终裁；即使 FAIL 轴全翻也不产生发档动作（ensemble 是新预测器，须走完整 R-1
  gate 另案送审）；Arm B 任一 FAIL 轴翻 PASS → 先报 REV 复核，再下任何结论。

## 四、边界重申

1. 本裁定仅针对 (a)-v2 单一预注册实验；(b) 止损（78a3f17）与 (c) V6 裁定（a236546）
   不受影响，三件互不耦合。
2. 本运行不产生发档、rec-probe、提交动作；rec-probe user-gated 维持。
3. rpe1 单轴否决结构不变；G3 对冻结单模型的终裁（aaf52c4 + 327b5ce）不变。
