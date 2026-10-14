# ml v2.1-(a)-v2 seed-ensemble LOCO —— 运行结果纪要（判定权在 REV）

- 日期：2026-10-14；撰写：ml-expert；状态：**240 训练按预注册运行完毕（exit 0，
  CPU ≈35 分钟），判定如实执行：豁免路径不通（jurkat 否决轴不可豁免），无 FAIL→PASS
  翻轴；结果无论方向全部入档（C3），判定权在 REV**
- 运行：`scripts/xgb_seed_ensemble.py`（spec `ml_v21av2_seedensemble_spec.md` §2–§3；
  REV 裁定 `rev_v21av2_seedensemble_spec_review.md` C1–C4）
- 逐折数据：`outputs/xgb_amp/loco_seed_ensemble.json`（git add -f，C1）——Arm A 逐折
  6 轴 × 6 split_seed × 5 成员 rmse + Arm B 逐折全量 × {SET_A, SET_B}
- E1 清单冻结执行：SET_A={0,7,13,42,2026}、SET_B={1,2,3,4,5}、SPLIT_SEEDS=
  {101,202,303,404,505,606}，运行中未加 K/未换 seed/未加 split

## 1. C1 判定重算表（供 REV 从逐折记录独立重算）

口径（C2 锁死）：σ_split_ens 只从 Arm A 六个 ensemble rmse 取（ddof=1）；delta_ens 取
Arm B SET_A（全量）；门槛 = 2×σ_split_ens；子采样位移不入判定；delta_frozen 仅作冻结
单模型对照值不进判定（REV §一）。本纪要落稿前已从逐折记录独立重算全部 σ 与判定，
与 json axes 块逐轴一致（§2）。

六轴全表（Arm B SET_A）：

| 轴 | g1 | rmse_ens | delta_ens | σ_split_ens | 2σ | 判定 | delta_frozen（对照） | SET_A−SET_B 位移 |
|---|---|---|---|---|---|---|---|---|
| gwps | 0.7151 | 0.3635 | −0.3516 | 0.05556 | 0.11112 | PASS | −0.3518 | −0.0303 |
| k562_ess | 0.3353 | 0.2896 | −0.0457 | 0.00227 | 0.00455 | PASS | −0.0417 | −0.0004 |
| **rpe1** | 0.7599 | 0.7833 | **+0.0234** | 0.01185 | 0.02369 | **FAIL（名义 eligible）** | +0.1177 | −0.0865 |
| **jurkat** | 0.4152 | 0.5712 | **+0.1560** | 0.04199 | 0.08398 | **FAIL（不可豁免 = 否决轴）** | +0.0794 | +0.1313 |
| hepg2 | 0.5370 | 0.4239 | −0.1131 | 0.00354 | 0.00709 | PASS | −0.0861 | −0.0489 |
| **hct116** | 0.5675 | 0.6602 | **+0.0927** | 0.04658 | 0.09316 | **FAIL（名义 eligible）** | +0.0828 | +0.0586 |

逐 FAIL 轴判定重算（|delta_ens(Arm B, SET_A)| vs 2×σ_split_ens，ddof=1）：

| FAIL 轴 | \|delta_ens\| | 2×σ_split_ens | 裕量 | 名义 eligible |
|---|---|---|---|---|
| rpe1 | 0.02340 | 0.02369 | **+0.00029** | true |
| jurkat | 0.15600 | 0.08398 | **−0.07202** | **false（否决）** |
| hct116 | 0.09270 | 0.09316 | **+0.00046** | true |

**判定结果**：豁免结构要求三个 FAIL 轴全部 eligible；jurkat 不可豁免 → 豁免路径不通，
无论 rpe1/hct116 名义 eligible 与否。三 FAIL 轴全部仍为 delta_ens > 0，**无 FAIL→PASS
翻轴**（C4"先报 REV 复核"触发器未击发）。

## 2. 管线忠实性（入档证据，全部通过）

1. **σ_split_ens 独立重算命中**：从 Arm A 逐折 rmse_ens（每轴 6 个）重算 ddof=1，
   六轴 σ 与 json axes 块逐轴一致（gwps 0.05556 / k562_ess 0.00227 / rpe1 0.01185 /
   jurkat 0.04199 / hepg2 0.00354 / hct116 0.04658）。
2. **Arm B 成员 0（seed 0，全量）== 冻结锚**：rpe1 0.8776 / jurkat 0.4946 等逐轴与
   `loco_frozen.json` rmse_model 一致（|Δ|<5e-5，json 记录 4dp）——全量单成员即冻结
   模型复现，管线忠实。
3. **g1 锚无漂移**：Arm B rmse_g1 逐轴 == loco_frozen.json rmse_g1（断言通过）。
4. **逐折字段齐备**（REV 预检 §二.4 清单）：Arm A 含 split_seed/member_seeds/
   member_rmses(5)/member_mean/rmse_ens/n_train/rmse_g1/delta_ens；Arm B 另含 seed_set；
   增量写盘与 /tmp/seed_ens_run.log [sum] 行逐轴一致。

## 3. C 条款执行

- **E2 预测器声明（逐字绑定，供未来 gate_report 引用）**：本判据对应预测器 =
  seed-ensemble(SET_A={0,7,13,42,2026}, K=5, 逐样本 log 空间算术均值
  `np.mean(preds, axis=0)`, 同 FEATS(39 维)/PARAMS/ROUNDS=600/CLIP)。单模型读数不得
  引用本方差；引用本豁免证据的任何未来 gate_report 其预测器必须逐字 = 上述定义（C3）。
- **E3 未放松**：乘数 2×/ddof=1/split-only 原生口径，ensemble 用自身 delta_ens 判定；
  delta_frozen（+0.1177/+0.0794/+0.0828）仅作对照列入表。
- **C2 跨臂口径**：σ 只从 Arm A（90% 子采样折）取；Arm A/B 间子采样位移不入判定。
  SET_A−SET_B 位移（n=2）只作不确定性声明、不作方向证据：rpe1 −0.0865、jurkat
  +0.1313、hct116 +0.0586——三个 FAIL 轴的 seed 集合位移均与门槛同量级或更大
  （rpe1 位移 0.0865 ≈ 3.7×门槛 0.02369），判定对 seed 集合选择敏感。SET_B 反事实
  （仅声明）：jurkat SET_B δ+0.0247 < 2σ 0.08398（名义 eligible）、rpe1 SET_B
  δ+0.1099 > 0.02369（不 eligible）——不构成任何方向证据。
- **C4 翻轴检查**：无任一 FAIL 轴翻 PASS（三轴 delta_ens 均为正），"任一翻 PASS →
  先报 REV 复核"触发器未击发；冻结锚 loco_frozen.json / loco_multisplit.json 未动；
  不重开 aaf52c4；本运行不产生发档 / rec-probe / 提交动作。
- **C3 判定权**：结果无论方向全部入档；豁免是否成立、后续动作与否，判定权在 REV。

## 4. 偏离预期披露（E4：无论方向入档）

spec §3 预注册预期为"rpe1 需 delta_ens < ~0.01 量级才可能过线，预期终局 FAIL；
jurkat 存在可信路径（seed 方差或可消位移）"。实测偏离两处，如实披露：

1. **jurkat 恶化，"可信路径"证伪**：ensemble delta_ens +0.156，较冻结单模型 +0.0794
   **近乎翻倍**（rmse_ens 0.5712 vs 冻结 0.4946）。Arm B SET_A 成员谱
   [0.4946(s0), 0.6373(s7), 0.7256(s13), 0.8922(s42), 0.6768(s2026)]——s0 即冻结锚且为
   最优成员，成员 rmse 均值 ≈0.685。log 空间逐样本均值后 ensemble 0.5712 介于最优成员
   与成员均值之间，但远差于冻结单模型：**jurkat 的 LOCO seed 效应是轴级相关固定位移
   （7909e94 §五 s7−s0=+0.178 同向实测的再确认），log 空间平均只消除不相关分量，
   消不掉共有的固定缺口**。换 seed 集合不能修 jurkat。
2. **rpe1 未达 spec 预期的 <~0.01 量级**：实测 +0.0234，名义 eligible 纯属裕量仅
   0.00029（阈值的 1.2%）的擦线；且 SET_A−SET_B 位移 −0.0865（SET_B 下不 eligible，
   0.1099 > 0.02369）显示该"名义 eligible"不稳健。成员层面 s13 离群（0.576，其余
   4 成员 0.861–0.878），rpe1 折对 seed 高度敏感。

对照 PASS 轴副效应（仅入档，不作方向证据）：hepg2 ensemble −0.1131 优于冻结 −0.0861
（改善 0.027）；gwps −0.3516 ≈ 冻结 −0.3518；k562_ess −0.0457 略优于 −0.0417。
seed 集合位移对 PASS 轴无害。

## 5. 结论

1. (a)-v2 预注册判定如实执行：**豁免路径不通**（jurkat |+0.156| > 2σ=0.08398，名义
   NOT eligible；三 FAIL 轴全 eligible 的必要条件不成立），且无任何 FAIL→PASS 翻轴。
2. rpe1/hct116 的名义 eligible 裕量（0.00029/0.00046）与集合位移（0.0865/0.0586）
   严重不成比例，属门槛擦线读数，不构成任何方向结论。
3. 冻结单模型 G3 FAIL 结论（rpe1 单轴否决，aaf52c4 + 327b5ce）不变；(a)-v2 不产生
   发档 / 提交 / rec-probe 动作；ensemble 作为新预测器若未来启用，须走完整 R-1 另案
   送审（C4 锁死）。
4. 机制收尾观察（仅证据素材）：seed-ensemble 在固定位移主导的轴（jurkat）上放大劣化、
   在 PASS 轴上无害或微益——与 (a) 案"轴级 seed 固定效应"结构一致，seed 平均不是
   LOCO FAIL 轴的修复通道。
