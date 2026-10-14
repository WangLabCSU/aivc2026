# REV 终裁：R-1 G1–G4 逐项裁决 + slot1 否决权表态（2026-10-14）

- 撰写：REV（红队独立裁决人）
- 依据：R-1 预注册 `docs/team/rev_R1_发档预注册.md`（commit f91d20c，锁定先于一切 gate 证据）
- 裁决对象：ml-expert `docs/team/gate_report.md`（commit aaf52c4，G1 FAIL / G2 PASS / G3 FAIL / G4 遵守）
  与 PI 证据包 `docs/team/pi_G1三臂对拍证据.md`（含 06:05 EDT 口径和解附录）
- REV 职权：仅否决权，不新增判据、不放松、不代行 PI 处置

## 0. 终裁摘要

| 项 | ml-expert | REV 终裁 | 说明 |
|---|---|---|---|
| G1 oracle-g 对拍 | FAIL | **维持 FAIL** | oracle 臂退化字面 FAIL + RAW 尺度 −1.7% < 5% + P_H1 未测 |
| G2 clip 饱和 | PASS | **维持 PASS** | g_report.json 全部数字独立核对一致 |
| G3 LOCO | FAIL | **维持 FAIL** | 3/6 轴劣化；豁免通道要求送审时方差，无 → 不成立 |
| G4 mse 口径 | 遵守 | **维持 遵守** | 无 affine 外推声明，合规 |
| slot1 = λ47（官方名 bizard-v0.2） | 预注册默认 | **不否决** | 预注册 FAIL 分支默认，本轮不行使否决权 |

**总执行指令：按预注册默认执行 —— XGB-MVP 不发档，10-14 唯一 slot 发 λ47，官方名 bizard-v0.2。**

## 1. 独立验证记录（REV 自行复核，非转抄）

1. **LOCO**（`outputs/xgb_amp/loco_frozen.json` 逐行核对）：gwps 0.3633/0.7151（−49.2%）、
   k562_ess 0.2935/0.3353（−12.4%）、hepg2 0.4509/0.5370（−16.0%）、rpe1 0.8776/0.7599（+15.5% 劣化）、
   jurkat 0.4946/0.4152（+19.1% 劣化）、hct116 0.6503/0.5675（+14.6% 劣化）；
   6/6 轴 rmse_model < rmse_const；pass_all_ctx=false。与 gate_report §3 表格逐数一致。
2. **G2 饱和率**（`outputs/xgb_amp/g_report.json`）：bench 96 靶 sat 0/0；submit A/B/C（289 靶）sat 0/0；
   train implied 1.52%/4.87%（hepg2 子集 0.31%/2.80%）；E[g] 四处 1.5465/1.7191/1.7009/1.7448/1.5756。
   与 gate_report §2 一致。
3. **四臂 scored.csv**（`outputs/metrics/local_bench/*/Q_HepG2_scored.csv` from_replicate 列均值重算）：
   overall 0.35209 / 0.36179 / 0.35020 / −0.31092，nmae 0.19954 / 0.29553 / 0.11305 / −3.92290，
   mse 四臂恒 0。与 PI 总表逐数一致。
4. **oracle ≡ uniform×3 退化**（bench_arms parquet 独立重构）：逐靶回归 rec delta ~ oracle delta
   反解隐含 g，rec 臂隐含 g_recenter 最小值 0.6058 与 `g_bench_hepg2.parquet` 的 g_recenter 最小值
   **逐位一致**；raw 臂隐含 g 均值 1.60 ≈ g_raw 1.5465（回归噪声内）；oracle/rec 逐靶 delta std 比
   均值 3.13、最大 4.95，与「oracle g 96/96 clip 至 3.0 + g_recenter ∈ [0.61, 1.94]」的期望分布吻合。
   三臂构造与申报一致，无未披露的额外加工。

以上四项全部通过。裁决建立在与撰写方独立复算一致的证据上。

## 1. G1 —— 维持 FAIL

R-1 原文判据：「P_H1 nmae 聚合增益 ≥5% 或 mse 可复现改善」，口径绑定探针 2（RAW 误差尺度，口径 B）。

1. **oracle 分支字面 FAIL，且退化机制独立证实**。oracle g = A/(16·sd) 对全部 96 覆盖靶 ≥4.205 →
   clip 触顶 → oracle 臂 ≡ uniform×3（≈本地 λ48）。§1 第 4 项验证即此机制。oracle 臂 nmae 两口径
   均灾难恶化（from_replicate −3.92291；RAW 0.8834→1.5390），未达成任何 ≥5% 增益。
   「绝对幅度缩放即使 oracle 级完美也摧毁 nmae」这一上限证明以最强形式否定了幅度方向。
2. **结构分支同样不达标**。即便按 PI 证据包第 3 条的宽松解读（判据对象是「g 结构有效」而非
   「oracle 臂达标」），在预注册的 RAW 尺度上最优臂（rec）增益 −1.7% < 5%，且在种子噪声带
   |Δ|≈0.04 内——与 REV 镜像先验（逐靶标量幅度校准 nmae 上限增益 ≤4%）相容。
3. **from_replicate +48.1% 不予采信**。该读数定义在 delta 得分尺度（anchor−实际型，正值=更好），
   与预注册的 RAW 误差尺度百分比不可互比。R-1 只许收紧不许放松，REV 不在裁决阶段引入新尺度；
   PI 附录第 3 条已自行撤回「rec 臂字面 PASS」解读，与本裁定一致。
4. **mse 分支不可用**。官方镜像 bench 四臂 expr_mse 恒 0——本地 λ16 幅度缩放连 oracle 都推不离 0，
   mse≠0 探针的本地预注册价值已死，无「可复现改善」可言。
5. **P_H1 未测**。预注册指定上下文未跑（Q_HepG2 替代已由 team-lead 接受为 deviation 备案），
   该缺口如实保留：G1 的判定建立在 Q_HepG2 单上下文证据上，判 FAIL 不需要 P_H1 数据，
   但判 PASS 将需要——缺口方向与 FAIL 结论一致。

**REV 终裁：G1 FAIL 维持。** 三重理由（oracle 退化 / RAW 尺度 −1.7% < 5% / P_H1 未测）任一独立成立。

## 2. G2 —— 维持 PASS

推理侧饱和 bench+submit 六表全 0/0；训练隐含 1.52%/4.87%（hepg2 子集 0.31%/2.80%），全部 ≤15%。
独立核对 `g_report.json` 逐数一致。clip[0.5,3.0] 参数化在本档保留，log 空间无界头列入 v2.1 非阻塞项。

## 3. G3 —— 维持 FAIL（豁免不成立）

loco_frozen.json：rpe1 +15.5%、jurkat +19.1%、hct116 +14.6% 三轴 rmse_model > rmse_g1（独立核对一致）。
R-1 判据为「任一 LOCO 轴点估计劣化 → 不发」，点估计层面三轴触发。唯一豁免通道要求
**送审时**已附该轴多重 split 方差且劣化 < 2× 轴间标准差——当前无任何方差数据，
事后补测按预注册无效。gwps/k562_ess/hepg2 三轴改善与 6/6 轴优于训练均值常数只能说明模型有信号，
不构成对「任一轴劣化即不发」条款的豁免。

**REV 终裁：G3 FAIL 维持。** 此项独立构成不发档依据，与 G1 无关。

## 4. G4 —— 维持 遵守

全部 mse/nmae 声明为 local 相对比较；无 affine 系数向官方换算；D-5 校准 gate 废除未启用；
本地 mse 恒 0 故无官方 mse 转换声明素材。合规。

## 5. slot1 = λ47（官方名 bizard-v0.2）—— 不否决

1. **预注册默认，锁定先于证据**。G1-FAIL → slot1 λ47 分支写定于 f91d20c（2026-10-13），
   先于本次三臂数据产生。REV 的职权是否决，不是在证据后重开参数选择。
2. **λ 曲线新证据不构成否决理由**。λ39 +0.0886 > λ55 +0.0330 表明峰值在 39–55 之间且偏向 λ39 侧，
   这削弱 λ47 中点的期望值，但恰也使 λ47 成为剩余未测点中信息量最大的探针；且该读数来自
   已发布的官方结果，属证据后知识，不能反过来改写预注册默认。若 team-lead 认为新证据足以
   改选 λ39 附近点，那是 PI 处置权的重启，不是 REV 否决权的行使对象。
3. **命名确认**：λ47 官方名 **bizard-v0.2**（版本号跟随提交序列，XGB-MVP 不发档故 v0.2 空出；
   此映射覆盖 a1312c4 与 gate_report §5 原文的 v0.3 绑定——与 team-lead 2026-10-14 处置及
   PI 附录第 6 条一致）。官网 model 名只写 bizard-v0.2，无任何后缀。
4. **操作警告（随档携带）**：`delta_submit_g_{A,B,C}.parquet` 按 g_raw（未重定心）缩放；
   λ47 发档不受影响，但任何未来 rec 臂提交必须先用 `g_submit_*.parquet` 的 `g_recenter` 列
   重缩放，直接提交现文件等于发 raw 臂（两口径均证 nmae 恶化）。

## 6. 技术备注（不改变 gate 结论，修正入档表述）

1. **「本地 nmae 预测 −0.174 与官方 −0.17433 完全一致」属 in-sample 闭环，非 out-of-sample 预测**。
   该 −0.174 来自家族 affine 拟合（nmae a=+0.02781, b=−0.02235），而拟合点本身就含 λ55 的官方读数：
   0.02781×(−5.463782)−0.02235=−0.1743，复现该点在代数上必然。诚实的验证表述是两条：
   (a) **方向一致性**——本地 nmae 对 λ 放大的响应与官方同号（λ16→λ55 本地 +0.19954→−5.46378 恶化，
   官方 λ39→λ55 −0.0168→−0.17433 恶化），这是 bizard 家族首个同向 local→official 证据；
   (b) **in-sample 闭环**——全指标 predict path overall 0.0396 vs 官方 0.0330，残差在家族拟合
   resid_sd 量级。后续引用该校准时按此口径写，不写「预测完全一致」。
2. **rec-probe 保持 user-gated**。gate_report 已将其定性为 PI 自由裁量项而非推荐项，
   REV 无补充意见；G1/G3 双 FAIL 的已记录风险由行使方自担，g_recenter 重缩放警告见 §5.4。
3. **v2.1 优先序附注**：多重 split 方差基建是 G3 豁免通道的唯一活门（送审时有效，事后无效），
   建议排 v2.1 首位，先于加权重训与 log 空间头。

## 7. 收束

G1 FAIL · G2 PASS · G3 FAIL · G4 遵守——四项全部维持 ml-expert 裁决，无推翻。
任一 FAIL 已独立封死 XGB-MVP 发档路径；slot1 按预注册默认发 λ47，官方名 bizard-v0.2。
**本轮 REV 不行使否决权。**
