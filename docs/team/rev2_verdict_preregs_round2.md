# REV 平行复审裁定 — rev_prereg_bizard_v06_dl_delta + rev_prereg_target_prior_v1（第二轮）

- 日期：2026-10-15；裁定人：rev-reviewer2（独立复审，team-lead 代收入档）
- 对象：`docs/team/rev_prereg_bizard_v06_dl_delta.md`（修订版）、`docs/team/rev_prereg_target_prior_v1.md`（修订版）
- 初审（REVISE）全部修订项已于 2026-10-15 应用；本轮为平行复审。

## 裁定

**整体裁定：两份均 PASS。**

- v0.6 预注册即具备开训前置（§8 硬前置满足，3-5 天预算与 v0.5 保底回退自洽）；
- target-prior v1 进入 LOPO ≥+2pp 硬前置等待，门过方可接入、门不过自动顺延；
- 训练与接入期间按两文 §6/§8 冻结纪律执行，本 REV 不再接受训练后任何门槛调整请求。

## 复审确认要点（摘要）

Doc A（v0.6 DL Δ̂）：
- C2 g 冻结关系澄清到位（L22 + L55，冻结项作用于已发布链，v0.6 线内无幅度头不冲突）；
- basal_rep 作用域已限定为新增 DL 成员内部条件化表征；
- 门槛 1-3 基线与口径钉死（含 sha256 冻结纪律，开训时锁定、不得更换）；
- λ47=+0.0456 数字核正；6.3×/3.6× 算术验证（0.3187/0.0502≈6.35，0.3187/0.0886≈3.60）；
- encoder-init 择优指标与验证集训练启动时锁定一次，定性明确。

Doc B（target-prior v1）：
- 冻结锚前移至"任何本地评估计算之前"（cap + basal 证据→幅度映射先冻结）；
- 门 1/2 基线钉死（现行冻结生产链 λ47 谱系 + 现行 affine，六上下文冻结 artifact）;
- 池过滤在 A.1 源加权之前，与 gene_weights (source,gene) 键（3920033，作用于加权期内）作用点不同、互不替代、无重叠；
- 与冻结 affine 次序已声明（先验注入在 affine 之前）;
- 跨文档范围闭合：机制仅作用于现行 linear-skip 线性链，v0.6 DL Δ̂ 落地后不自动迁移，须重跑 §4 门 + 另案复审；
- §6 冻结条款全项收录（9 项）。

## 非阻塞附注（nice-to-have）

- 两文的门 1/门 2 基线 artifact 建议在门槛计算时同步记录 sha256 入档（不影响本裁定效力）。

## 效力

- 初审 REVISE 关闭；第二轮 PASS 终局；
- v0.6 开训解锁（§8 硬前置满足）；target-prior 等待 LOPO 门（cb_v22_c1_freeze.md §8），门过即接入、不过自动顺延；
- 本 REV 不再接受训练后任何门槛调整请求。
