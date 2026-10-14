# C2 XGB 幅度头 MVP — gate_report (R-1 预注册判据逐项裁决)

日期: 2026-10-14。模型: xgboost reg:pseudohubererror, depth=7, lr=0.05, 600 rounds,
subsample/colsample 0.8, seed=0; 39 维冻结特征 (F-1); y=log A (top-200 |logFC| 均值);
g = clip(exp(ŷ)/(16·sd_v1), 0.5, 3.0)。LOCO/训练/g 表三阶段脚本
`scripts/xgb_amplitude_head.py` 全绿; 产物在 `outputs/xgb_amp/`。

裁决总览: **G1 FAIL · G2 PASS · G3 FAIL · G4 遵守(仅局部相对口径)**
→ **slot1 不发 XGB-MVP, 按预注册默认发 λ47 备选 (bizard-v0.3)**;
XGB-MVP 保留为"预注册校准探针"候选(见 D-5), 由 PI 裁量占用次日额度。

---

## R-1 G1 — oracle-g 对拍: FAIL

三臂 bench (Q_HepG2, --lam 16, RAW, 基线 bzbiz10skip_lam16):

| 臂 | pds | mse | jac | nmae | fid | reach |
|---|---|---|---|---|---|---|
| lam16 基线 | 0.7012 | 0.0189 | 0.0755 | 0.8834 | 0.5955 | 0.5054 |
| lam55 对照 | 0.7214 | 0.1055 | 0.0754 | 1.7840 | 0.5851 | 0.5743 |
| xgbamp_rec (g_recenter, E[g]=1) | 0.7059 | 0.0182 | 0.0752 | 0.8682 | 0.5868 | 0.4874 |
| xgbamp_raw (g_raw, mean≈1.55) | 0.7152 | 0.0255 | 0.0762 | 0.8972 | 0.5935 | 0.5282 |
| oracle3 (96 靶 oracle g 全饱和→×3) | 0.7209 | 0.0820 | 0.0756 | 1.5390 | 0.5878 | 0.5701 |

- **oracle-g 无增益**: 96 覆盖靶的 oracle g 全部 ≥4.205 → clip 上限 3.0,
  oracle 臂退化为 uniform ×3 ≈ 本地 λ48。与 λ55 官方结论一致:
  放大→pds 微升 (+0.0197) 但 **nmae +0.6556 / mse +0.0631 恶化主导**。
  oracle nmae 天花板为负 (oracle3 nmae 比 lam16 差 74%), 远低于 REV 镜像
  probe-2 的 oracle-k 参照 (+0.2%/+4.0%) 与 G1 门槛 +5%。
- **主臂 xgbamp_rec**: nmae −0.0153、mse −0.0007 (相对基线 −3.7%)、pds +0.0046,
  六指标无 >0.02 的恶化; 但 nmae 增益 1.5% 远 < 5% 且在种子噪声带内
  (anchor_v11_sweep 四 seed 束 overall 跨度 −0.0808…+0.0228, nmae ~0.6;
  本地 Δ<~0.04 一律视为噪声)。**G1 判据 (nmae 增益≥5% 或可复现 mse 改善) 不满足 → FAIL**。
- **止损检查 (REV nmae-系标签显式止损)**: 主臂无此问题 (nmae 改善且 pds/fid 同向或
  噪声级); oracle 臂命中: 15/64 靶 pds 改善但 nmae 恶化>0.01。
- **跨指标 g 冲突率 (镜像 probe-2 口径, 64 有效 nmae 靶, 3 臂逐靶最优)**:
  nmae 最优臂与 pds 最优臂一致 37.5%, 与 fid 一致 14.1%, 三指标全一致 **1.6%** —
  与 REV probe-2 的跨指标 k 一致率 0.0 同构, 幅度轴的"最优"不可跨指标迁移。
- **逐靶 nmae**: 主臂 improved 24 / worsened 11 / 平 29 (median Δ −0.001),
  方向净正但幅度在噪声带内。

## R-1 G2 — clip 饱和: PASS

- bench 预测 g: sat_lo=0, sat_hi=0 (n=96); A/B/C 提交侧全部 0/0 (n=289 each)。
- 训练隐含 g (A/(16·sd), 全 24,398): sat_lo 1.52%, sat_hi 4.87% — 远低于 10–15% 重参数化线。
- 分布: bench g_raw mean 1.5465 / median 1.4724 / IQR [1.20, 1.74];
  A/B/C g_raw mean 1.70–1.74, IQR ≈ [1.60, 1.86], 近似全局常数 ×1.7
  (重定心后 submit sd 仅 0.11, bench sd 0.283 — 逐靶离散存在但弱)。

## R-1 G3 — LOCO 相对判据: FAIL (如实报告, 无豁免证据)

| ctx | rmse_model | rmse_g1 | Δ vs g1 | rmse_const |
|---|---|---|---|---|
| gwps | 0.3633 | 0.7151 | **−49.2%** | 0.5101 |
| k562_ess | 0.2935 | 0.3353 | −12.4% | 0.5141 |
| rpe1 | 0.8776 | 0.7599 | **+15.5%** | 1.0983 |
| jurkat | 0.4946 | 0.4152 | **+19.1%** | 0.6423 |
| hepg2 | 0.4509 | 0.5370 | −16.0% | 0.8575 |
| hct116 | 0.6503 | 0.5675 | **+14.6%** | 0.7270 |

- 模型 vs 训练均值常数: 6/6 胜; vs g=1 零假设 (v1 现状): 仅 3/6 胜。
- rpe1/jurkat/hct116 相对劣化 14.6–19.1%, 远超任何噪声带; pass_all_ctx=false。
- 决策相关的 hepg2 (bench ctx) 双轴均胜 (−16.0% / −45.5%), 但 G3 是全轴判据。
- 复审时无 multi-split 方差证据可提供 (未跑重复划分), 豁免条款不启用。

## R-1 G4 — mse 声明口径: 遵守

- mse 仅以本地 RAW 相对差表述 (主臂 −0.0007, 噪声级; oracle 臂 +0.0631 恶化)。
- 无任何 affine 官方外推。本地 λ→mse 线性插值参考 (lam12–55 五点): λ47 ≈ 0.087,
  仅供 slot2 预期管理, 不入档为预测结论。

---

## REV 逐项回应

### A-1 (oracle-g 对拍 + 停止条件) — 触发
oracle 臂 ≡ uniform ×3 (96/100 饱和); RAW 上 nmae +0.6556、mse +0.0631,
与 λ55 官方 (nmae −0.1743 恶化主导, overall 回退) 方向一致。
**A-1 停止条件成立: oracle-g 对拍未买到 mse/nmae, C2 按 R-1 降级, slot1 → λ47。**
补充镜像: 跨指标 g 冲突率 1.6% (≈probe-2 的 0.0), oracle 头寸不存在。

### A-2 (g 分布 + 饱和率) — 报告如上
G2 PASS (全表 0% 饱和)。已知 caveat: 训练侧 A/(16·mean_abs) 口径下 15.92% 超阈值
(前次分析), 换 sd 口径后降至 1.5%/4.9% — 提示 g 标定的主要不确定性在 s_v1 的
sd/mean_abs 口径差, 不在模型。

### B-1 (E[g]=1 重定心 + 家族偏移敏感性) — 已执行
- 主臂即重定心臂: bench g_recenter ∈ [0.606, 1.940], sd 0.283, 均值恒 1。
- **家族偏移 −0.30 敏感性结论**: g_raw 均值 1.55 (bench) / 1.72 (提交) 显著>1,
  且训练隐含 g 中位数 1.43 — 头部学到的主体是"全局幅度上调 ≈×1.5–1.7",
  与其说是逐靶信号, 不如说是在补偿 v1 λ39 与签名幅度之间的家族级偏移
  (λ55 官方回退已证该方向在官方轴不成立)。重定心后剩余逐靶离散 (sd 0.11–0.28)
  才是真正的 C2 信号, 而本 bench 显示其增益 (nmae −0.015) 在噪声带内。
- **E[g]=1 敏感性检查 (team-lead 问)**: 若不重定心直接发 (xgbamp_raw 臂),
  等效全局 λ ≈ 16×1.55 ≈ 24.7 — 落在本地 λ 曲线 mse 拐点 (λ18–20 之后) 之外,
  RAW mse +0.0066 / nmae +0.0138 恶化, pds +0.0139 微升。**E[g]=1 约束是必要的**;
  家族偏移假设得到支持 (g 头大部分变异 = λ/家族偏移补偿)。

### D-5 (affine gate 废除) — 已执行
未用 official_alignment predict 做任何 gate 决策; 判据全部换为 LOCO 相对 + RAW 对拍
+ 预注册 G1–G4。首发若发, 定性为"预注册校准探针": mse≠0 官方点本身即产出
(17 连发 mse 恒 0)。当前主臂重定心后幅度变化 [0.61, 1.94]×, mse≠0 概率未定
(官方 mse 轴可能对 <2× 幅度变化不敏感), 探针价值需如实标注"不确定"。

### F-1 (特征冻结) — 已执行
39 维在 LOCO 前冻结, LOCO 只验收 (loco_frozen.json), 无任何搜索/调参;
未动 α/λ 标量以外的超参。特征重要性 (gain): log_absmean 55.9%, fp_hepg2 12.3%,
log_sd 5.0%, fp_gwps 5.0% — 前 60.9% 集中在幅度先验 (log_absmean+log_sd),
fp_* 为 ctx 级偏移; hepg2 内部判别力来自 s_v1 系特征, 结构合理但解释了
LOCO 在 rpe1/jurkat/hct116 的失效 (缺 ctx 专属幅度先验)。

### pseudohuber 大幅值欠权重 (REV 建议) — 分位数校验
标签 log A 分布的隐含 g (median 1.4345, IQR [1.03, 1.94]) 与预测 g 分布
(bench median 1.47) 对齐, 未出现大幅值系统性截断; 但 in-sample rmse 0.2043 与
LOCO 0.29–0.88 的差距提示高幅度靶残差主导。**未做 A 加权重训** (F-1 冻结期内
禁止架构改动), 列入 v2.1 提案。

### 团队-lead λ55 追问补充
- **g(λ39) vs g(λ55) 对照臂**: g 头是在 λ39 (本地 16) 尺度上标定的;
  xgbamp_raw 臂 (g 不重定心) 等效 λ≈24.7, 介于两控制点之间, RAW 表现亦居中 —
  g 头没有提供超出"全局 λ 重标定"的增量。
- **mse 预测区间**: 本地主臂 mse Δ=−0.0007 (±种子噪声 0.002 级);
  官方映射无有效 mse 校准 (D-5), 仅可声明"官方 mse 是否≠0 不可预测, 探针性质"。

---

## slot2 判断 (R-1 默认路径)

- **slot1 (bizard-v0.2, XGB-MVP): 不满足发档判据** — G1/G3 FAIL。
  若 PI 仍想占用 slot1 发 XGB 探针 (校准价值), 建议发 **xgbamp_rec 臂**
  (B-1 重定心版, 本地六轴无 >0.02 恶化, 风险受控), 命名 bizard-v0.2;
  这是预注册性质的偏差决策, 需 PI 明示。
- **slot2 (bizard-v0.3): λ47** (λ39–55 峰值区间中点, 本地 λ 曲线
  pds/reach 单调升、nmae 拐点 λ18–20, λ47 为 pds 增量与 nmae 损耗的折中)。
  不同家族原则不适用于 bizard 内部 λ 微调 (REV E-4 限制的是跨家族连发)。
- 本地 bench 种子噪声证据: anchor_v11_sweep 4 seed 束 overall −0.0808…+0.0228,
  nmae 跨 ~0.6 — 主臂全部 Δ 在噪声带内, 不构成任何方向性结论的额外证据。

产物: scripts/xgb_amplitude_head.py; outputs/xgb_amp/{loco_frozen.json,
model_final.json, feature_importance.json, g_report.json, g_bench_hepg2.parquet,
g_submit_{A,B,C}.parquet}; bench 臂数据 data/interim/xgb_amp/bench_arms/,
指标 outputs/metrics/local_bench/xgbamp_{xgbamp_rec,xgbamp_raw,oracle3}_lam16/。
