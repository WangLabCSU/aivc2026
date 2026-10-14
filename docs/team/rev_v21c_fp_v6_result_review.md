# REV 复核裁定 — v2.1-(c) V6 运行结果（认可终局关闭生效）

- 日期：2026-10-14；裁定人：REV；对象：`docs/team/ml_v21c_fp_v6_result.md` +
  `outputs/xgb_amp/fp_v6_loco.json`（ml-expert 撰写/运行）；裁定送达 ml-expert 后入档。

## 一、独立重算全部通过（fp_v6_loco.json × loco_frozen.json 锚 × 纪要三方对读）

1. **冻结臂逐位 == 锚**：六轴 rmse_frozen_rerun 与 loco_frozen.json rmse_model
   float `==` 全轴成立（0.363337 / 0.293534 / 0.877556 / 0.494641 / 0.450927 /
   0.650290）——管线忠实确认，纪要 §1.3 断言实测为真。
2. **逐轴 delta 独立重算**：V6Δ = gwps −0.3466 / k562_ess −0.0409 / rpe1 +0.0147 /
   jurkat +0.0775 / hepg2 −0.1384 / hct116 +0.1187；frozenΔ = −0.3518 / −0.0417 /
   +0.1177 / +0.0794 / −0.0861 / +0.0828 —— 与纪要 §2 表逐字命中，json 内部 delta
   自洽 |Δ|<5e-5。
3. **NN 锚命中**：五值（0.981920×2 / 0.724137 / 0.796601 / 0.719042 / 0.683803）
   match_rev=true 全轴；对称性签名 symmetry_train_max 与 nn_values 逐位相等
   （如 rpe1 0.7241370616631331）——"测试侧路由与 frozen 逐位同"的实测确认。

## 二、裁定：认可 (c) 终局关闭生效

- FAIL 轴 = {rpe1, jurkat, hct116} 与冻结版同集、无翻轴、pass_all_ctx=false →
  预注册主判据命中，C2 方向义务执行正确：fp 适配空间四类全证伪声明入档，
  此后不开 fp 变体，REV 侧同表锁死。
- C3 seed 声明认可成立：rpe1 seed 位移 ≈0（(a) 案 C2 实测一致）点估计即结论；
  jurkat/hct116 s7 固定效应只会加深 FAIL；本运行无 FAIL→PASS 读数，C3 高风险
  情形不存在。
- C4 全 PASS 分支未触发，无义务。纪要 §5 备案观察（训练侧 self 分布轴特异、
  效应异向）认可为 (c) 证据链闭环素材；§4.1 网格搜索禁令与不做清单维持。
- G3 终裁（aaf52c4 + 327b5ce）不变；冻结锚 loco_frozen.json 未动（逐位比对为证）。

## 三、a932848 竞态备案

已由 142b281 空提交补记自行处理，V6 三文件（纪要/json/脚本）已入档且 message 层
读数可检索，REV 侧不再要求任何补录动作。

## 四、(a)-v2 待办提醒（裁定原文）

240 训练完成后 loco_seed_ensemble.json（逐折）+ 纪要入档报 REV。REV 将独立重算
σ_split_ens（ddof=1，仅 Arm A 六个 ensemble rmse）与逐 FAIL 轴
|delta_ens(Arm B, SET_A)| vs 2×σ_split_ens；C2 跨臂口径（子采样位移不入判定、
SET_A−SET_B n=2 仅作声明）与 C4（任一 FAIL 轴翻 PASS 先报 REV 复核）纪律照裁定执行。
