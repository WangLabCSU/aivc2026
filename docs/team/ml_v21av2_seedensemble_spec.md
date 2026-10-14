# ml (a)-v2 seed-ensemble 评测 —— 预注册 spec（E1–E4 逐条映射）

- 日期：2026-10-14；撰写：ml-expert；状态：**预注册送审稿，未运行任何本 spec 实验**
- 任务来源：team-lead（56db89d 复核 memo + 排队指令）「seed-ensemble 评测：每折对 K≥5 个
  xgb seed 的预测取均值后再算 rmse，σ_seed 压到 σ_seed/√K，让 std 真正由 split 方差主导；
  seed 均值化本身可能直接改善 jurkat/hct116 的 rmse……spec 先行 → REV 认可 → 再跑」
- 判据框架：REV 裁定稿 7909e94 §四 E1–E4 + 保留项 split-only 口径；R-1 只许收紧不许放松。
- 与 (a)(b)(c) 关系：(a) 方差基建已入档但豁免通道三轴全部落空（REV 保留项行使，7909e94）；
  本 spec 是该落空的唯一预注册修复路径。(b) 已止损、(c) spec 已另行送审（3610d97），无耦合。

## 0. 动机（承接 (a) 的两个已入档事实）

1. (a) 42 折分解：jurkat σ_seed 0.1258 ≫ σ_split 0.0310；hct116 σ_seed 0.0733 ≫ σ_split
   0.0164——合并 std ~80% 是 seed 分量，触发 REV 保留项，split-only 口径下三轴豁免落空。
2. seed 效应是轴级系统位移（s7 相对 s0：jurkat +0.178 / hct116 +0.104 / hepg2 −0.066，
   轴内方向一致）。ensemble 均值化对**交换分量**按 ~1/√K 压缩，对**固定位移分量**则部分
   抵消（不同 seed 的预测误差在样本级去相关时，均值预测的 rmse 低于成员均值）——后者
   是经验性主张，不由本 spec 假设，由 Arm B 直接实测。

目标双问句：(i) K-seed ensemble 能否把豁免判据的 std 真正还原为 split 方差（名实相符）；
(ii) ensemble 点估计本身是否改善 FAIL 轴（提交链潜在免费增益）。

## 1. 设计（E1/E2 逐条落实，先冻结后运行）

**E1 seed 清单（先冻结，运行前不改、不加、不换）**：

| 清单 | 值 |
|---|---|
| 成员 seed 集 SET_A（K=5） | {0, 7, 13, 42, 2026} |
| set-level 稳健性集 SET_B（仅 Arm B 用，K=5） | {1, 2, 3, 4, 5} |
| split_seed（Arm A，6 个） | {101, 202, 303, 404, 505, 606} |

- SET_A 含 0/7：与 (a) 全量 s0/s7 读数连续可对照；13/42/2026 为无观测史新 seed。
- SET_B 与 SET_A 不相交，专测 ensemble 级 seed-set 位移（n=2 差值，仅作不确定性声明量级，
  不作方向证据——C4 同款纪律）。
- E1 纪律：运行中不加 K、不换 seed、不加 split；若判据不过，加 K/改口径须另案预注册交 REV。

**K=5 论证**：交换分量按 σ_seed/√K 压缩，K=5 给 jurkat 0.1258/√5≈0.056、hct116
0.0733/√5≈0.033——量级已与 (a) 的 σ_split 同阶；且 (a) 实测 seed 效应以**固定位移**为主
（非纯交换噪声），均值化的实际压缩率由 Arm B 实测裁定而非本公式。K=5 总成本 240 训练
≈35 分钟；K=10 加倍仍可行，但按 E1 纪律升级须另案送审，不在本运行内追加。

**Arm A（豁免方差基建）**：6 轴 × 6 split_seed × K=5 成员 = 180 训练。
- 每 split：与 (a) 完全同协议（`default_rng(split_seed)` 按 5 训练 ctx 分层等比抽 90%，
  行级 (target,ctx) 抽样——同 regime 理由沿用 REV bba2a7a 焦点 (i) 裁定）；
  5 个成员模型**共享同一 keep 索引**，仅 xgb seed 不同 → 折内差异纯为 seed 分量。
- 每折产出：5 个成员 rmse + ensemble rmse（**E2 定义：逐样本预测算术均值后算 rmse**，
  y=log A 尺度，与冻结 LOCO 同 split 同口径）；delta_ens = rmse_ens − rmse_g1。
- 轴级汇总：σ_split_ens = std(6 个 ensemble rmse, ddof=1)（C1 同款）+ 逐折记录供 REV
  独立重算（C2 同款）：split_seed、5 成员各自 rmse、member mean、ensemble rmse、g1、delta。

**Arm B（点估计 + seed-set 不确定性）**：全量训练（无子采样）LOCO 6 轴 × 2 seed 集 × K=5
= 60 训练。SET_A 为主判据读数；|rmse_SETA − rmse_SETB| 逐轴入档为 ensemble 级 seed
不确定性声明（满足 REV §五"LOCO 读数必须附 seed 不确定性声明"）。

**E2 预测器声明（预注册锁死）**：本证据所 gate 的预测器自单模型变为
**seed-ensemble（SET_A，K=5，逐样本均值）**。未来任何引用本豁免证据的 gate_report，
其实际提交预测器必须与本定义逐字一致（同 FEATS/PARAMS/ROUNDS/CLIP/K/seeds/均值方式）；
单模型读数不得引用本方差。

## 2. 预注册判定（先于运行锁死）

- **豁免主判据（逐 FAIL 轴）**：|delta_ens(Arm B, SET_A)| < 2×σ_split_ens(Arm A, ddof=1)
  → 该轴豁免名义可用。rpe1 单轴即否决（R-1"任一轴劣化→不发"结构不变）。
- **E3 遵守（不放宽）**：乘数固定 2×、ddof=1、split-only 为原生口径（本设计中 seed 分量
  已被均值化压入折内，不再存在 (a) 的合并 vs split-only 名实之争——这正是设计目的）；
  预算结构不放宽、不上调乘数、不换锚。三轴原 delta_frozen（0.1177/0.0794/0.0828）仅作
  冻结单模型对照值继续入档；ensemble 用其自身 delta_ens 判定（E2 预测器已变更，
  沿用旧 delta 反而名实不符——此为 E2/E3 的联合忠实读法，报 REV 备案）。
- **预期读数（非判据，防解读漂移的事前声明）**：rpe1 需 delta_ens < ~0.01 量级才可能
  过线，预期终局 FAIL（单轴否决维持，G3 对冻结单模型的终裁不受影响）；jurkat 存在
  可信路径（需 delta_ens < ~2×σ_split_ens，若 σ 量级与 (a) 的 σ_split 0.031 相当，
  即需 ensemble 改善 ≥0.02）；hct116 需改善 >0.05，难度显著更高；hepg2/gwps 为
  PASS 轴观察项（hepg2 s7 系更优，均值化可能稀释其优势——如实记录，不作判据）。
- **E4**：结果无论方向全部入档（`outputs/xgb_amp/loco_seed_ensemble.json` + 纪要）；
  判定权在 REV。
- **C3 延续**：本运行不重开 aaf52c4 的 G3 终裁。即使 FAIL 轴全翻，也不产生任何发档动作
  ——ensemble 是新预测器，须走完整 R-1 gate（G1 nmae/G2 clip/G4 口径）另案送审。

## 3. 明确不做（预注册锁死）

1. 不做 K 的网格搜索（不加 K、不换 seed 集重跑择优）；判据不过时仅报分解，升级与否交 REV。
2. 不改 FEATS/PARAMS/ROUNDS/CLIP；不动 loco_frozen.json / loco_multisplit.json 锚。
3. 不产生发档、rec-probe、提交动作；rec-probe user-gated 维持。
4. 若 Arm B 与事前预期矛盾（任一 FAIL 轴翻 PASS）：先报 REV 复核（C1/C2 同款纪律），
   再下任何结论；单读数必附 SET_A/SET_B 位移声明。

## 4. 成本与产物

- 240 次训练（180 + 30 + 30），按 (a) 实测单折 ~9s 约 35 分钟，后台分轴续跑式
  （复用 xgb_multisplit.py 的增量写盘骨架，新脚本 scripts/xgb_seed_ensemble.py）。
- 产物：outputs/xgb_amp/loco_seed_ensemble.json（逐折 + decomp）；
  结果纪要 docs/team/ml_v21av2_seedensemble_result.md；均入档后报 REV/team-lead。

## 附：配置（可复现）

- 数据：data/interim/xgb_amp/features_full.parquet（24,398 行，39 维冻结 FEATS，y=log A）
- PARAMS：逐字同冻结版（pseudohuber / depth7 / lr0.05 / 600r / sub0.8 / col0.8 / hist /
  n_jobs=40）；成员 seed ∈ SET_A / SET_B；ROUNDS=600；FRAC=0.90
- ensemble 均值在 log 空间逐样本算术平均；rmse/g1 口径与冻结 LOCO 完全一致
- E1–E4 对应：E1=§1 清单表；E2=§1 声明 + §2 判定绑定；E3=§2 不放宽条款；E4=§2/§3 入档义务
