# 双源两发预登记（提交前落盘）

## 第 1 发 daily-best：dualsrc λ3.5
- 假设：hct116 源的 nmae 增益（本地 +0.059）在官方兑现，pds 代价不超预期 → overall ≈ +0.025~0.03
- 预期读数：overall ∈ [+0.020, +0.035]；nmae ∈ [-0.005, +0.02]（vs λ3.5 单源 +0.017 持平或略优）；pds ∈ [0.30, 0.36]
- 决策规则：overall > +0.0247 → 双源转正为 daily-best 主线；∈ [+0.015, +0.0247] → 平手区，保留单源为默认并封存双源；< +0.015 → 双源弃用

## 第 2 发 calibration：dualsrc λ4.5
- 假设 H2：官方 fid(λ) 斜率线性外推（本地已知 λ4.5 nmae 崩塌——正检验"官方斜率 vs 本地斜率"背离）
- 预期读数：若线性 fid(4.5) ≈ -0.16、overall ≈ +0.03；若本地崩塌兑现 fid(4.5) ≈ -0.6、overall < 0
- 决策规则：overall > λ3.5 → 官方斜率线性，λ 阶梯续爬（λ5.5）；overall < λ3.5 → λ 阶梯封顶于 3.5，主力转向 Tahoe 药扰辅助 + fusion v3

## 修订（10/2 晚，REV 终审后）：浅版 hct116 签名致命 bug 使原预登记失效

Expert E 发现浅版 hct116_all.parquet 构建器致命 bug（靶码未按 list 长度展开→全靶签名近同）。
原预登记的 dualsrc λ3.5/λ4.5 均基于退化签名，前提失效。REV 终审（agent_930b28bb）裁定：

- **第 1 发 daily-best：dualsrc-deep λ3.5**（B，20261003_dualsrcdeep_lam35.vcc）——修复版深度签名，
  bench Q 归一化 0.318（非 in-sample 最高）。预期 official +0.03~+0.06（nmae 通道按 per-metric 折算 ≈+0.034）。
- **第 2 发 calibration：fusion v3 @λ2.5**（C，20261003_l15fusion_v3.vcc）——架构探针：
  S_HCT116 in-sample +0.151 打折后预期 official -0.10~+0.02；重点读 fid/pds 方向。
- λ4.5 阶梯检验延后（需用深度签名重建后再发）。
- 判读规则沿用：第 1 发 vs +0.0247 定双源去留；第 2 发 fid/pds 方向定 v3 保守化路线。

## 二次修订（Expert E 交付 fusion v3.5 后）：第 2 发改 fusion v3.5

- fusion v3.5（tahoe 药扰辅助多任务，w=0.3+has_sig）：LOCO-HepG2 **0.3313** vs v3 0.3214（+0.0099 过 gate 未稀释）；
- **pds 0.658 反超 v2（0.617）**——直击官方 pds 短板（fusion v2 官方 pds 0.151）；
- 资产纠错：basal_sw480 实为 CVCL_0546（SW620 转移株）已注明；
- **第 2 发改 fusion v3.5 @λ2.5**（取代 fusion v3——LOCO 更高且 pds 修复）；fusion v3 保留为消融记录。

## 10/4 窗口预登记（终版）

- **第 1 发 daily-best：dualsrc-deep λ4.5**（20261003_dualsrcdeep_lam45.vcc）
  - 假设：官方 fid(λ) 斜率延续（四点 -0.838/-0.378/-0.199 → 预期 λ4.5 ≈ -0.10~-0.15）
  - 预期：overall **+0.05~+0.08**（pds 惯性 +；nmae 双源持平）
  - 判读：>+0.0506 → λ 阶梯转正主线续爬；∈[+0.0247,+0.0506] → 平手区封顶观察；<+0.0247 → λ 封顶 3.5
- **第 2 发 architecture：bizard-v0.1 群飞聚合**（20261003_bizard_v01.vcc）
  - 定位：fusion 家族官方兑现独立性检验（历史上限 +0.048/fusion v2）
  - 预期：-0.10~+0.02（家族偏移 -0.42±宽；in-sample 打折）
  - 判读：>-0.0168（fusion v3.5）→ 群飞聚合官方成立 → v0.2 加码；< fusion v2 → 架构回退启发式+λ
- 备选（若明晨需要替换）：bz02dense 聚合 Δ̂ 的 A/B/C 重推理版（6 鸟去毒种群，Q 0.4085）
