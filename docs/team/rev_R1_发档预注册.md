# REV R-1 预注册：XGB-MVP slot1 发档审查判据（先于 gate_report 锁定）

日期: 2026-10-13 | 审查人: REV | 性质: **预注册**——以下判据在看到 ml-expert gate_report 之前锁定并 commit，事后不得放宽；若判据本身有漏洞，只允许新增更严判据。

## 触发条件

ml-expert 产出 XGB-MVP（C2 幅度头）gate_report 后，slot1 发档前由 REV 独立审查，本文件即审查契约。

## 判据（G1–G4 全部 PASS 才推荐发档）

**G1 oracle-g 对拍**（口径复用 `docs/team/rev_state_st_probe_补充.md` 探针2：逐靶事后 oracle、同 local bench、同 6 指标）：
- 主轴: P_H1 上下文 nmae 聚合增益 ≥5%，或 mse 有可复现改善；
- **FAIL 条件**: nmae 增益 <5% 且 mse 无改善 → C2 降级「不推荐发档」，slot1 转发 λ 备选（λ39–55 之间；默认建议 **λ47**，理由: λ39/λ55 两端已测，中点信息增益最大，且不与已发布 λ55 同质）；
- 证据要求: 须同时报 nmae 上限与跨指标 g 冲突率（nmae 最优 g 与 fid/pds 最优 g 的逐靶一致率），镜像探针2口径。

**G2 g 分布健康度**：
- g 预测经 clip[0.5,3.0] 后饱和率（落在任一边界 ±1e-3 内的比例）≤15% → PASS；
- 饱和率 >15% → 发档须附 g 重参数化方案（如 log 空间无界头 + 后验 clip），否则降级。

**G3 LOCO 相对判据**：
- 任一 LOCO 轴点估计劣化于 v1 冻结基线（同 split、同口径）→ 不发；
- 预注册故意不设噪声容忍带；唯一豁免通道: gate_report 在**送审时**已自带该轴的多重 split 方差，且劣化幅度 < 2× 轴间标准差——事后补测方差无效。

**G4 mse 声明验证（D-5 已知盲区）**：
- 背景: affine 校准的官方锚点 7/7 全部 mse=0（见 `docs/team/rev_v2_红队评审.md` item D）→ mse 轴 local→official 映射斜率不可辨识；
- 若 gate_report 声称 mse≠0 或以 official mse 预测为发档依据，REV 核验其依据:
  (a) mse 改善是否基于 local 相对比较（λ55 同上下文 local bench 对照）→ 有效；
  (b) 若直接由 affine 外推 official mse → 该声明作废，不得作为发档依据；
- mse 结论一律以 local 相对差值呈现，禁止 official 换算。

## 判定流程

1. 收到 gate_report → 逐条 G1–G4 核验证据链（引用文件+行级数值）；
2. 输出裁决（SendMessage + 入档）: 每条判据 PASS/FAIL + 证据；
3. 任一 FAIL → 「不推荐发档」+ 指定 slot1 替代物（按 G1 FAIL 默认 λ47；其他 FAIL 由 team-lead 决定替代物，REV 只给否决）；
4. 本契约只约束 slot1；slot2（λ 备份）不受影响。每日 2 发 UTC 额度与 one-in-progress 槽位约束由 team-lead 把握（见 memory: vcc-submission-ops）。
