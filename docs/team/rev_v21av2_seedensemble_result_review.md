# REV 裁定 — v2.1-(a)-v2 seed-ensemble 240 训练结果（豁免裁定：不成立，终局）

- 日期：2026-10-14；裁定人：REV；对象：commit 08de172（ml-expert 结果纪要
  `ml_v21av2_seedensemble_result.md` + `outputs/xgb_amp/loco_seed_ensemble.json`
  + `scripts/xgb_seed_ensemble.py`）
- 裁定：**豁免不成立（终局）**——REV 从逐折记录独立重算全部 σ 与判定，与纪要 C1
  表逐轴命中；jurkat 否决轴 |+0.15600| > 2σ=0.08398，三 FAIL 轴全 eligible 的必要
  条件不成立；无 FAIL→PASS 翻轴，C4 触发器未击发；G3 终裁（aaf52c4 + 327b5ce）不变。

## 一、独立重算（全部通过，逐轴逐位）

从 `loco_seed_ensemble.json` 48 折（36 sub + 12 full）独立重算：

1. **E1 清单冻结确认**：Arm A 每轴恰 6 折、split_seed ∈ {101,202,303,404,505,606}、
   member_seeds == {0,7,13,42,2026}；Arm B 每轴 2 折 seed_set ∈ {SET_A, SET_B}，
   SET_B 成员 == {1,2,3,4,5}；48 折 × 5 成员 = 240 训练与声明一致，运行中未加 K/
   未换 seed/未加 split。
2. **σ_split_ens 独立重算逐轴命中**（ddof=1，每轴 6 个 sub 臂 ensemble rmse）：
   gwps 0.05556 / k562_ess 0.00227 / rpe1 0.01185 / jurkat 0.04199 / hepg2 0.00354 /
   hct116 0.04658——与 json axes 块及纪要 §2 逐轴一致（|Δ|<5e-6）。
3. **delta_ens（Arm B SET_A）独立重算命中**：−0.3516 / −0.0457 / +0.0234 / +0.1560 /
   −0.1131 / +0.0927，与 axes 块逐轴一致。
4. **逐 FAIL 轴裁定重算与纪要 §1 表逐字命中**：rpe1 |0.02340| < 0.02369 裕量
   **+0.00029**；jurkat |0.15600| > 0.08398 裕量 **−0.07202（否决）**；hct116
   |0.09270| < 0.09316 裕量 **+0.00046**。
5. **逐折内部自洽零失败**：全部 36 sub 折 member_mean == mean(member_rmses)、
   delta_ens == rmse_ens − rmse_g1（|Δ|<6e-5），ens_rmses_per_split 与逐折 rmse_ens
   一致；无一处转录漂移。
6. **冻结锚逐位确认**：Arm B 成员 0（seed 0 全量）逐轴 == `loco_frozen.json`
   rmse_model（4dp 全等：gwps 0.3633 / k562_ess 0.2935 / rpe1 0.8776 / jurkat 0.4946 /
   hepg2 0.4509 / hct116 0.6503）；g1 锚逐轴无漂移。管线忠实四证全过，与纪要 §2 一致。

## 二、豁免裁定（C3 判定权行使）

1. **豁免不成立（终局）**。预注册结构要求三 FAIL 轴全部 eligible；jurkat
   |+0.15600| > 2σ=0.08398，NOT eligible（否决轴）→ 必要条件不成立，豁免路径不通，
   无论 rpe1/hct116 名义 eligible 与否。ml-expert 判定如实，予以确认。
2. **rpe1/hct116 名义 eligible 不构成任何方向结论**。裕量 0.00029/0.00046（阈值之
   1.2%/0.5%）与集合位移 −0.0865/+0.0586 严重不成比例（rpe1 位移 ≈ 3.7× 门槛）；
   SET_B 反事实下 rpe1 +0.1099 > 0.02369 不 eligible、jurkat +0.0247 < 0.08398 名义
   eligible——判定对 seed 集合选择敏感，擦线读数不入任何方向证据（C2 跨臂口径维持）。
3. **无 FAIL→PASS 翻轴确认**：三 FAIL 轴 delta_ens 均为正，C4"任一翻 PASS → 先报
   REV 复核"触发器未击发。
4. **E4 偏离预期披露认可**：两处披露如实。jurkat"可信路径"证伪接受——成员谱
   [0.4946(s0，即冻结锚且最优), 0.6373(s7), 0.7256(s13), 0.8922(s42), 0.6768(s2026)]
   显示 log 空间逐样本平均消不掉轴级共有固定位移，rmse_ens 0.5712 较冻结单模型
   0.4946 恶化 +0.077、delta_ens +0.156 较冻结 +0.0794 近乎翻倍，与 7909e94 §五
   s7−s0=+0.178 轴级固定效应结构互证；rpe1 未达 spec 预期的 <~0.01 量级，s13 成员
   离群（0.576，其余 0.861–0.878），名义 eligible 属门槛擦线不稳健。
5. **机制收尾观察采纳为证据素材**：seed-ensemble 在固定位移主导轴（jurkat）放大
   劣化、在 PASS 轴上无害或微益（hepg2 改善 0.027 / gwps / k562_ess 持平或微益）——
   与 (a) 案"轴级 seed 固定效应"结构一致，seed 平均不是 LOCO FAIL 轴的修复通道。
   仅入档为素材，不另开变体线。

## 三、边界与后续

1. **G3 终裁（aaf52c4 + 327b5ce）不变**：冻结单模型 rpe1 单轴否决维持；(a)-v2 不
   产生发档 / 提交 / rec-probe 动作。
2. **C4 维持**：冻结锚 loco_frozen.json / loco_multisplit.json 未动（本次复核以逐位
   比对为证）；不重开 aaf52c4；ensemble 作为新预测器若未来启用，须走完整 R-1 另案
   送审，本 (a)-v2 方差证据不得直接移植（E2 逐字绑定条款维持：未来 gate_report 引用
   本判据时预测器须逐字 = seed-ensemble(SET_A={0,7,13,42,2026}, K=5, 逐样本 log 空间
   算术均值 np.mean(preds, axis=0), 同 FEATS(39 维)/PARAMS/ROUNDS=600/CLIP)）。
3. **资产定位**：loco_seed_ensemble.json 判定素材地位与 loco_multisplit.json 同级
   （方差基建证据，不作发档依据）。
4. **LOCO 修复线状态更新**：(a) 方差豁免、(a)-v2 seed-ensemble、(b) 特征归因、
   (c) fp 适配四线全部负结果闭环。此后 LOCO 修复不再开无新机制证据的变体；任何新线
   须先出机制假说送 REV 预审再动工。

## 四、附：成员层结构性观察备案（team-lead 传达 PI 侧，REV 已逐折独立验证）

三条观察均从 48 折逐折记录复现，仅入档为证据素材，不改变 §二 裁定，不触发任何行动：

1. **jurkat Arm A 双峰**：s101/s606 逐折 delta_ens +0.2346/+0.2336，其余四 split
   +0.1499~+0.1550；s101/s606 的最优成员（s0）rmse 0.3949/0.3957 好于冻结 0.4946，
   ensemble 反而 0.6498/0.6488 更差——最优成员偏离群包时 log 空间逐样本平均放大位移，
   与 §二.4"轴级共有固定位移"裁定互证。
2. **rpe1 s13 结构性离群**：六个 split 全部如此（逐折 0.5428–0.6091，跨 split 均值
   ≈0.579），其余四 seed 跨 split 均值 0.858–0.872（逐折范围 0.8435–0.8884）——
   spec 预期"rpe1 需 <~0.01"未达的机制根源；s13 离群是 split 稳健的结构性现象，
   非单折噪声。
3. **hct116 s606 +0.0112 擦线近过**：6 折中唯一非 FAIL 级位移，其余 5 折
   +0.0753~+0.1414；单 split 擦线不构成翻轴证据，与 §二.2"裕量 vs 集合位移严重
   不成比例"裁定一致。
4. 复核补充：Jensen 型检查 rmse_ens ≤ member_mean 全 48 折 0 违反（PI 侧复核读数，
   REV 独立复算确认）。
