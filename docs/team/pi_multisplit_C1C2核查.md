# PI 独立复核 — 42-fold multi-split C1/C2 预检与 σ 分解（2026-10-14）

对象: `outputs/xgb_amp/loco_multisplit.json`（42 折 = 6 轴 × 3 split × 2 seed + 6 FULL 对照，
ml-expert v2.1-(a) 产出）。以下为 PI 在转交 REV 前的独立重算。

## C1（ddof=1）✓ 全部通过

6 轴 std_axis 均与 ddof=1 重算一致（ddof=0 会系统性低估 ~8–10%，json 用的是对的）。

## C2（σ_seed / σ_split 分解）— 结构性发现

对每轴 3×2 网格做均值分解（split 均值间 std → σ_split；seed 均值间 std → σ_seed）：

| 轴 | σ_seed | σ_split | std_axis(json) | 主导源 |
|---|---|---|---|---|
| gwps | 0.0956 | 0.0529 | 0.0988 | seed |
| k562_ess | 0.0013 | 0.0031 | 0.0036 | split（两者皆极小） |
| rpe1 | 0.0021 | 0.0053 | 0.0080 | split（两者皆极小） |
| jurkat | **0.1258** | 0.0310 | 0.1042 | **seed** |
| hepg2 | 0.0468 | 0.0019 | 0.0364 | **seed** |
| hct116 | **0.0733** | 0.0164 | 0.0747 | **seed** |

## 裁定含义：当前运行支撑不了 G3 豁免

REV 保留的 re-adjudication 口径是 split-only。按 2×σ_split 门槛重裁：

- jurkat: δ=0.0794 > 2×0.0310=0.0620 → **不合格**（json 里 eligible=True 被翻转）
- hct116: δ=0.0828 > 2×0.0164=0.0328 → **不合格**（同上翻转）
- rpe1: 任何口径下 δ=0.1177 ≫ 噪声（σ 合计 0.0057）→ 退化真实，维持不合格

即：当前 std_axis 里 ~80%（jurkat/hct116）是 xgb seed 噪声，不是 split 方差——
正是 ml-expert 预警的"名实不符"情形，且三个可豁免候选轴全部落空。

## 修复建议（转 ml-expert 评估成本）

(a)-v2: seed-ensemble 评测 —— 每折对 K≥5 个 xgb seed 的预测取均值后再算 rmse
（σ_seed 降到 σ_seed/√K），使 std 真正由 split 方差主导。42 折 × 5 seed 量级可控。
在此之前，本 json 只作基建验证，不进任何 gate_report 豁免裁定。

## FULL 对照（seed7, 与既有单 seed 结果一致）

gwps 0.3993 / k562_ess 0.2970 / rpe1 0.8644 / jurkat 0.6373 / hepg2 0.3832 / hct116 0.6345
（jurkat 0.6373 即此前引用的"纯 seed 噪声"值；rpe1 FULL 0.8644 vs g1 0.7599 → δ≈0.105，与 +0.1177 同量级）

## 与 ml-expert 结果纪要的收敛（2026-10-14 补）

ml_v21a_multisplit_result.md 已读：split-only std（jurkat 0.0310 / hct116 0.0164）与
本 memo 重算同值；其口径更严谨——split 均值各含半个 seed 方差，故 split-only std 是
**高估**，本 memo 的 σ_split 是下界，两者夹住真值，结论（split-only 下 jurkat/hct116
FAIL）在两种口径下均成立。三轴豁免落空的判定以 ml-expert §3 + 本 memo 为共同依据。

附：ml-expert §5.2 已被 PI 采纳为送审要求——未来任何单 seed LOCO 读数必须附带
seed 不确定性声明（本运行实测轴级系统位移量级 ±0.07~0.18：jurkat +0.178 /
hct116 +0.104 / hepg2 −0.066，方向轴内一致 = 固定效应而非交换噪声）。
