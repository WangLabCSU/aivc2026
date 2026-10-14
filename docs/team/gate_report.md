# C2 XGB-MVP（Bizard v2.0）发档 gate 报告 —— docs/team 正式入档版

- 日期：2026-10-14
- 撰写：ml-expert（C2 执行人）
- 依据：R-1 预注册 `docs/team/rev_R1_发档预注册.md`（2026-10-13 锁定，先于本报告 seen）
- 证据：`outputs/xgb_amp/`（g_report.json / loco_frozen.json / feature_importance.json）、
  `outputs/metrics/local_bench/{bzbiz10skip_lam16, xgbamp_xgbamp_rec_lam16, xgbamp_xgbamp_raw_lam16, xgbamp_oracle3_lam16}/Q_HepG2_scored.csv`、
  `data/interim/xgb_amp/bench_arms/*_Q_HepG2.parquet`、PI 证据包 `docs/team/pi_G1三臂对拍证据.md`
- 详细技术版（五臂全表、REV 条目逐条回应、v2.1 提案）见 `outputs/xgb_amp/gate_report.md`（commit abcd991）；
  本文件是 R-1 要求的裁决前 canonical 入档版。

## 0. 证据范围与双口径声明（读数前必读）

1. **bench 上下文是 Q_HepG2，不是 P_H1**。R-1 G1 预注册原文写「P_H1 ctx nmae 聚合增益」——
   **P_H1 口径的 oracle-g 对拍＝未测**（P_H1 无 v1 pred delta 基建，三臂跑在 Q_HepG2 上）。
   下文 G1 以 Q_HepG2 证据裁决，并如实标注该缺口。
2. **scored.csv 存在两列口径**，方向一致但百分比不可互比：
   - **from_baseline**（RAW 层差分，镜像现行 +0.0886 口径）：nmae 基线 0.8834，rec 0.8682
     （Δ −0.0153，相对 −1.7%）；本地 bench 种子噪声底 |Δ|≈0.04，该量级≈噪声。
   - **from_replicate**（官方口径近似，PI 证据包采用）：nmae 基线 0.19954，rec 0.29553，
     即 rec 相对基线 **+48.1%**（该列是「anchor−实际」型得分，正值=更好）。
   - REV 镜像先验「逐靶标量幅度校准 nmae 上限增益 ≤4%」定义在 RAW 误差尺度上；
     from_replicate 的 +48.1% 是 delta 得分尺度上的百分比，与 ≤4% 先验**不可比**。
3. **mse 口径**：官方镜像 bench（expr_mse_unbiased_capped_norm）四臂全为 0；
   旧局部管线 RAW agg 曾给 lam16 mse≈0.0189 / oracle3 0.0820（口径不同，以官方镜像为准）。

## 1. G1 oracle-g 对拍 —— **FAIL**

R-1 原文判据：「P_H1 nmae 聚合增益 ≥5% 或 mse 可复现改善」。逐条对拍：

### 1.1 oracle 臂自身：灾难失败（两口径一致）

oracle g 取真值 A/(16·sd)，但 96 个覆盖靶**全部 ≥ 4.205 → 触顶 clip → 全等于 3.0**，
oracle 臂退化 ≡ uniform ×3 ≈ 本地 λ48。后果（Q_HepG2，from_replicate）：

| 臂 | overall | nmae | reach | pds |
|---|---|---|---|---|
| oracle3 | **−0.31092** | **−3.92291** | +0.1059 | +0.0562 |
| 基线 | 0.35209 | 0.19954 | — | — |

RAW 口径同向：nmae 0.8834 → 1.5390（+0.6556 恶化）。**绝对幅度缩放即使 oracle 级完美也摧毁 nmae**，
仅方向类指标受益。oracle 臂未达成任何 ≥5% nmae 增益 → 判据的 oracle 分支字面 FAIL。

### 1.2 三臂对照的信息量（结构验证完成，但不是 G1 的通过条件）

| 臂 | nmae from_replicate | nmae RAW | 解读 |
|---|---|---|---|
| rec（重定心 g，E[g]=1） | 0.29553（**+48.1%** 得分改善） | 0.8682（−1.7%，噪声带内） | 逐靶相对结构有效 |
| raw（原始 g，E[g]≈1.55） | 0.11305（−0.0865 恶化） | 0.8972（+0.0138 恶化） | 全局幅度校正有害 |
| oracle3 | −3.92291 | 1.5390 | 绝对幅度灾难 |

rec delta 宽度仅为 oracle 的 1/3.3（std 0.0111 vs 0.0362），per-target nmae 24 改善 / 11 恶化 / 29 中性。
**E[g]≈1.5–1.7 的含义**：g_raw mean bench 1.5465、submit A/B/C 1.72/1.70/1.74、train implied 1.5756——
四处一致说明幅度头学到的主要信号是 **v1 λ39 家族的全局幅度偏移补偿**，不是细粒度逐靶结构；
重定心后残余逐靶离散度很小（submit sd 0.11，bench sd 0.28）。raw 臂 ≈ 全局 λ24.7，两口径均恶化 nmae，
反证 E[g]=1 约束必要，也支持 REV 的 family-offset 假设。

### 1.3 mse 分支：本地不可用

四臂 expr_mse 恒 0——λ16 本地幅度缩放**连 oracle 都推不离 0**，mse≠0 探针的本地预注册价值已死
（官方 mse≠0 与否未知）。mse 分支不可复现改善 → 不满足。

### 1.4 逐靶 g 冲突率（镜像 probe-2 报告义务）

nmae-vs-pds 最优臂一致 37.5%、nmae-vs-fid 14.1%、三指标一致 1.6%（probe-2 镜像为 0.0）——
g 的逐靶方向与方向类指标高度冲突，与 1.2 的臂级观察一致。

### 1.5 结论

G1 **FAIL**，三重理由：(a) 判据绑定的 oracle 臂退化为 uniform×3 且 nmae 灾难恶化（两口径）；
(b) 即便放宽到「g 结构」本身，在镜像先验可比的 RAW 尺度上最优臂增益 −1.7% < 5%
（与 REV 镜像先验 ≤4% 相容）；(c) 预注册指定的 P_H1 上下文 **未测**。
R-1 只许收紧不许放松：rec 臂 +48.1%（from_replicate）不能作为 G1 PASS 依据——该读数
不在预注册尺度上，且 PI 亦不自行放宽，交由 REV 按预注册原文认定。
按预注册 FAIL 分支：**C2 降级，slot1 → λ47 默认**。

## 2. G2 clip 饱和率 —— **PASS**

- 推理侧饱和：bench（n=96）sat_lo/sat_hi = 0/0；submit A/B/C（n=289×3）均 0/0。
- 训练隐含 g（n=24,398）：sat_lo 1.52% / sat_hi 4.87%；hepg2 子集 0.31% / 2.80%。
- 全部 ≤15% 判据 → PASS，无需 log 空间无界头重参数化（列入 v2.1 提案，非本档阻塞）。

## 3. G3 LOCO —— **FAIL（豁免不成立）**

loco_frozen.json：pass_all_ctx = **false**，3/6 轴劣化于 g=1 基线：

| ctx | rmse_model | rmse_g1 | Δ |
|---|---|---|---|
| gwps | 0.3633 | 0.7151 | −49.2% |
| k562_ess | 0.2935 | 0.3353 | −12.4% |
| hepg2 | 0.4509 | 0.5370 | −16.0% |
| **rpe1** | **0.8776** | **0.7599** | **+15.5% 劣化** |
| **jurkat** | **0.4946** | **0.4152** | **+19.1% 劣化** |
| **hct116** | **0.6503** | **0.5675** | **+14.6% 劣化** |

6/6 轴优于训练均值常数（模型确实学到了信号），但 R-1 判据是「任一 LOCO 轴点估计劣化 → 不发」，
且**唯一豁免通道**要求 gate_report 送审时已自带该轴多重 split 方差、劣化 < 2× 轴间标准差——
当前无任何多重 split 方差数据（事后补测无效，PI 与 team-lead 均认定豁免不成立，本报告无方差可附）。
→ G3 **FAIL**。

## 4. G4 mse 声明口径 —— **遵守**

全部声明均为 local 相对比较；无任何 affine 系数向官方 λ39 的外推换算；D-5 affine 校准 gate
已废除未使用；official mse 转换声明一律缺席（本地 mse 恒 0，无声明素材）。合规。

## 5. 总裁决 —— **不推荐发档（XGB-MVP 不上 slot1）**

> **命名更正（2026-10-14, PI 处置裁定）**：本节及下文"λ47=bizard-v0.3"绑定作废。
> 版本号跟随实际提交序列——XGB-MVP 不发档故 v0.2 空出，**λ47 作为首个 plain 命名提交
> 取官方名 bizard-v0.2**。本文件正文保留原记录，以下按此更正解读。

G1 FAIL · G2 PASS · G3 FAIL · G4 遵守。R-1 verdict flow：任一 FAIL → 不推荐发档，
且 G1-FAIL 分支预注册了确定默认：**slot1 → λ47 备选**（官方名按提交序列为 bizard-v0.2，见上更正）。

理由压缩成三句：oracle 臂退化使 G1 的上限证明没有做成，而镜像先验尺度上 g 结构的可见增益
（−1.7%）低于 5% 门槛；LOCO 三轴劣化且无送审时方差可豁免；两处 FAIL 均触发预注册的
不发档分支，λ39/λ55 已双实测、λ47 为中点最大信息量替补——10-14 仅剩 1 发，按预注册发 λ47
是唯一不新增未验证风险的选项。

**slot 建议**：
- 10-14 唯一 slot → λ47（本地 interp overall ≈ 0.087），官方名 **bizard-v0.2**
  （提交序列规则, 见第 5 节更正; 命名规则：只写 bizard+版本）。
- XGB-MVP（rec 臂）**不占用该 slot**；若 PI 愿意在后续窗口把它作为预注册校准探针
  （检验官方 mse 是否 ≠0、官方 nmae 口径是否复现 rec 结构收益），需自担 G1/G3 双 FAIL 的
  已记录风险——这是 PI 自由裁量项，不是本报告推荐。
- **操作警告**：`delta_submit_g_{A,B,C}.parquet` 是按 **g_raw（未重定心）** 缩放的；若任何情况下
  要发 rec 臂，必须先用 `g_submit_*.parquet` 的 `g_recenter` 列重缩放，直接提交现文件等于发 raw 臂
  （bench 已证 raw 臂 nmae 恶化）。

## 6. v2.1 方向（非阻塞，供后续迭代）

1. A 值加权重训（pseudohuber 无系统截断证据，但 top 幅度端欠拟合可见）。
2. log 空间无界幅度头替代 clip[0.5,3.0]（G2 通过，故非阻塞）。
3. sd vs mean_abs 口径差（A-2：4.9% vs 15.9%）需在 v2.1 特征评审中定性。
4. 多重 split 方差基建先行——G3 豁免通道只在送审时有效，事后补测无效。
