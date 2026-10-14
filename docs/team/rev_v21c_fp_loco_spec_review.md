# REV 裁定 — v2.1-(c) fp 特征 LOCO 适配预注册 spec（V6 认可动工，附 C1–C4）

- 日期：2026-10-14；裁定人：REV；对象：`docs/team/ml_v21c_fp_loco_spec.md`（commit 3610d97，
  ml-expert 撰写）；裁定送达 ml-expert 后入档。
- 裁定：**认可 V6 按预注册动工**（6 次训练，分钟级），附 C1–C4 条款。预期主结果
  rpe1 仍 FAIL → (c) 终局关闭出负结果纪要；判据已由 spec §3 先于运行锁死，REV 予以确认。

## 一、独立复核（全部通过，逐轴逐位）

对 `outputs/xgb_amp/fp_loco_probe_v0.json` × `outputs/xgb_amp/loco_frozen.json` ×
`data/interim/xgb_amp/features_full.parquet` 三方独立重算：

1. **frozen 臂六轴与冻结锚逐位相同**：0.8775561451911926 / 0.494641 / 0.650290 /
   0.450927 / 0.363337 / 0.293534（与 loco_frozen.json folds 的 rmse_model 直接比较，
   float 相等）。探针管线忠实成立，与 spec §1 声明一致。
2. **V2_clampself 与 frozen 六轴全部 bitwise 相同**（float 直接比较 `==`）——spec §2
   "结构性定理"（xgb hist 分裂阈值 ≤ 训练 max ⇒ 测试值 ≥ 训练 max 走同一末箱）的实测
   签名成立。"测试侧 fp_self=1.0 OOD 误路由"假说证伪确认，测试侧重映射类适配判死。
3. **spec §1 表 24 个读数逐项重算一致**（|Δ|<5e-5）：6 轴 × frozenΔ/V1/V2/V3 全部命中；
   g1 列与锚 folds 的 rmse_g1 一致。无一处表抄错或转录漂移。
4. **V6 定义数据侧独立验证**：从 features_full.parquet 直接重算各轴测试行
   `fp'_self = max_{j≠t} fp_j(t)`——rpe1 0.724137（spec §3 声明 0.7241 ✓）、jurkat
   0.7966、hct116 0.6838、hepg2 0.7190、gwps/k562_ess 0.9819，各轴测试行为逐块常数，
   构造正确。信息通道与冻结版同类（ctx 级 basal 相关标量），无新泄漏面。

## 二、机制备注（加固预期主结果，非异议）

相关度对称性 ⇒ V6 测试行 self 值恒等于该列训练侧 max（四轴实测均如此）⇒ 按 V2 定理
其测试侧路由与 frozen 逐位相同 ⇒ **V6 全部效应只能经训练侧 self 列分布变化
（1.0 → NN 相关度）改变树分裂实现**。与 V1 读数（整列删除对三 FAIL 轴仅 ≤0.013 微动）
合读，"rpe1 仍 FAIL" 为合理预期主结果；若结果相反（任一 FAIL 轴翻 PASS），按 spec §4
披露义务先报 REV 复核，不自行下结论。

## 三、C 条款（随认可生效）

- **C1 产物入档**：V6 逐轴 json（逐轴 rmse_V6 / rmse_g1 / delta）git add -f 入档（同
  fp_loco_probe_v0.json 款）；结果纪要附 `_build_ctx_feats` self 分支实现 diff 或逐字
  公式落位，供 REV 独立重算。
- **C2 方向义务**：任一轴 FAIL（预期主结果）→ (c) 终局关闭，负结果纪要送 REV 入档；
  fp 适配空间四类（测试侧钳位 / 去自特征 / 去全族 / 语义对齐）全证伪声明随纪要落档，
  此后不开 fp 变体。
- **C3 单 seed 声明**：seed=0 读数凡用于结论（尤其任何 FAIL→PASS 方向），须附 7909e94
  §五轴级 seed 不确定性声明（实测位移 jurkat +0.178 / hct116 +0.104 / hepg2 −0.066 /
  gwps +0.135，方向轴内一致 = 固定效应）。
- **C4 全 PASS 分支确认**：spec §3 自带门槛（seed=7 复核 + 自带 6 模型 multi-split
  方差，再走 REV gate_report）为 REV 前置义务；届时方差口径沿用 (a) 案 split-only
  re-adjudication 与 REV 保留项。spec §4 四条不做清单（不网格搜索 / 不改冻结锚 /
  不重开 aaf52c4 / 披露义务）由本裁定一并锁死。

## 四、边界重申

1. 本裁定仅针对 (c) V6 单一预注册实验；(a)-v2 E1–E4 与 (b) 止损裁定（7909e94）不受影响。
2. (c) 产出仅为证据纪要，不产生发档动作；G3 终裁（aaf52c4 + 327b5ce）不变。
3. 冻结锚 loco_frozen.json 未被探针或 V6 运行修改（本次复核以锚逐位比对为证）。
