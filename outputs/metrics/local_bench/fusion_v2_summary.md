# fusion v2 汇总：HCT116 入池 + λ 扫描选型（2026-10-02 夜间窗口）

env: aivc2026 / L20 GPU / 64C。全部 bench 分数为 Q_HepG2 anchor-normalized（0=generic baseline，1=replicate），
官方标度经历史校准点偏移约 -0.21~-0.31（docs/本地→官方校准框架）。

## 1. fusion@λ 扫描（v1 all-5 Δ̂，Q_HepG2 in-sample）+ 启发式对照

| 预测器 | λ1.5 | λ2.0 | λ2.5 | λ3.0 |
|---|---|---|---|---|
| fusion v1 (in-sample) | 0.3270 | 0.3665 | **0.3879** | **0.3923** |
| 启发式 (g13b) | 0.2175¹ | — | 0.2668 | — |

¹ 历史 anchor_normalized.json（l11_lam1.5）。

fusion@λ2.5/3.0 关键分解：pds 0.540/0.537（启发式 0.745），fid **0.917/0.923**（启发式 0.730），reach 0.395/0.429（0.139）。
→ 学习式在高 λ 下全面反超启发式（+0.121 @2.5），fid 是主要增益来源；曲线 2.5→3.0 仅 +0.004（in-sample），
LOCO 上 +0.017（见下），斜率未闭合但减速。

## 2. HCT116 签名（data/interim/signatures/hct116_all.parquet）

- 源：data/public/xatlas/HCT116_3batches.h5ad，56,655 cells × 38,606 genes（去 12,962 个 ENSG 兜底 id），
  p53-WT 结直肠癌，NTC='Non-Targeting' 2,782 cells（3 batch 池化）。
- 15,450 非 NT perts 中 **≥5 cells 仅 4,075 perts**（每 pert 均值 ~3.5 cells，Perturb-seq 稀释型）。
  预注册 min_cells=5：51.2M 行 [target_gene, gene, logfc, n_cells]，竞赛 300 靶覆盖 56/300。
- 阈值依据（方向一致率 vs gwps，|logfc|≥0.25 层）：3-4 cells 0.579 / **5-9 cells 0.613** / ≥10 cells 0.537 → min5 落在最优层。
- p53/IFN 证据：NTC basal p53 模块 log1p-CPM 均值 **3.88**（K562 gwps 2.39；HepG2 3.90，RPE1 3.55）——补齐 K562 缺失的 p53-WT 语境；
  IFN 模块 2.01 为六上下文最高（context C=IFN-high 的补位）。TP53/MDM2 本身 <5 cells 未入集（CDKN1A 扰动 p53 模块 -0.047）。

## 3. fusion v2 LOCO-HepG2（训练池 {gwps,k562_ess,rpe1,jurkat,hct116}，hepg2 全排除）

| 模型 | λ1.5 | λ2.5 | λ3.0 |
|---|---|---|---|
| v0/v1 5-ctx LOCO | 0.2644 | 0.3076² | — |
| v2 6-ctx LOCO | — | **0.3222** | **0.3392** |

² 本次消融补跑（l15s1v0_fused_lam2.5）。分解 @λ2.5：λ 杠杆 +0.0432，hct116 纯增益 **+0.0146（+4.8%）**；合计 +0.058 vs v1@1.5。
fid：0.839（v2@2.5）/0.864（@3.0）vs v0@2.5 0.841；pds 0.617/0.622 vs 0.588。sig_hct116 特征 stage1 gain share 3.0%。

**重要修复（影响本轮全部 v2 数字）**：`common.load_basal` 精确匹配 'non-targeting'，
xatlas 的 'Non-Targeting' 大小写不符 → basal_hct116 全零静默失败（hct116 训练行 ~27% 的 basal 特征报废）。
已改为大小写不敏感 + 空集显式报错；修复前后 v2@2.5 0.3174→0.3222、@3.0 0.3302→0.3392。
A/B/C 提交特征不受该 bug 影响（basal 用 context_{A,B,C} 自身，v1 起即正常）。
另修 `extract_embeddings.tahoe` 对 dict 型 basal_h5ad 条目的崩溃 + 无 ensembl_id 列跳过。

## 4. A/B/C 提交物（v2 stage2 全 6 上下文重训，λ=2.5）

- `outputs/submissions/20261002_l15fusion_v2.h5ad`（4.25GB）+ `.vcc`（vcc prep 全绿）
- validation：360,000 × 18,533，entries 2.085B，entries/cell 5,791.6，density_margin 0.5611，integer_valued true
- Δ̂ sanity（vs v1 提交）：|Δ̂|>0.25 基因数/靶 mean A 24.6 / B 20.8 / C 14.2（v1: 7.1/6.9/5.9，×3.4），
  gwps 方向一致率 0.913-0.920（v1: 0.88-0.89），|Δ̂|mean 0.0174-0.0177
- metrics：`outputs/metrics/20261002_l15fusion_v2.json`；本汇总各级分数在 outputs/metrics/local_bench/{l15v1_fused_lam*,l15v2s1_fused_lam*,l15s1v0_fused_lam2.5,g13b_lam2.5_qhepg2}/anchor_normalized.json

## 5. 风险与建议

风险：
1. 官方从未见过 fusion@λ≥2：fid 官方斜率按启发式外推（local Δ→official Δ≈×2）不可靠，λ3.0 未获官方验证。
2. Δ̂ 富集度较 v1 放大 ~3.4×：若官方 fid 对幅度响应弱于本地（A/B/C 分布 ≠ HepG2），mse/nmae 端风险上升。
3. 选型仅基于 Q_HepG2 单上下文；context C=IFN-high 与 A=T 谱系的外推未验证（hct116 IFN basal 最高是利好旁证）。
4. S_HCT116 对 v2 是 in-sample，未用于选型；hct116 3-4 cells 层（8,662 perts）弃用，TP53/MDM2 直接证据缺席。

建议（明晨）：
1. 首发日更：`20261002_l15fusion_v2`（λ2.5，与官方最优 λ 锚点一致，LOCO 0.3222 vs v1 0.2644）。预期官方 overall 转正（校准偏移 -0.21~-0.31）。
2. 校准位：fusion v2 @λ3.0（LOCO +0.017）探 fid 斜率是否延续——仅需换 λ 重建载体，无需重训。
3. 若官方 fid 仍为瓶颈：下一步做 hct116 特征源扩容（sigstore 用 min3 层、训练标签保持 min5）与 context-C IFN 专项诊断。
4. 代码修复（load_basal 大小写 + tahoe dict 防护）建议 PI 复核后入 ledger；v1 提交（-0.0965）对应的 5-ctx 模型已被 v2 取代。

<!-- hct116-fulldepth:auto-begin -->
## 附: HCT116 全深度 fid 坍塌检验（夜间自动链, S_HCT116 全批次 ref）

- pooled: 3409169 cells / 109 batches / NTC 165777 /
  ≥30-cell perts 17358 (median 150.0);
  ref: 19218 cells, 100 targets (median 139.5 cells/靶)
- 归一化 fid 对比: **fusion v2@λ1.5 vs 启发式 λ1.5 = 0.7772 vs 0.6715
  (Δ=0.1057)**;
  fusion v2@λ2.5 = 0.8480; fusion v1(LOCO)@λ2.5 = 0.8533
- 结论: fusion 在 p53-WT CRC 上下文复现 fid 修复
- nmae 处置: available (unavailable: [])
- 官方 -0.838 型坍塌是否复现: 见启发式行 fid/overall 与 Q_HepG2(0.73/0.27) 对照

| 臂 | fid | pds | reach | jac | nmae | overall | anchor |
|---|---|---|---|---|---|---|---|
| heuristic_lam15 | 0.6715 | -0.6949 | -0.1385 | -1.0306 | 0.0909 | -0.1836 | bundle |
| fusionv1_lam25 | 0.8533 | 0.5090 | 0.3283 | -3.0175 | 0.1765 | -0.1917 | bundle |
| fusionv2_lam15 | 0.7772 | 0.0884 | 0.1023 | -0.7535 | 0.2052 | 0.0700 | bundle |
| fusionv2_lam25 | 0.8480 | -0.3357 | 0.2408 | -3.6184 | 0.1637 | -0.4503 | bundle |
<!-- hct116-fulldepth:auto-end -->

## fusion v3.5（Tahoe 药扰辅助任务）设计决策预案（2026-10-02，待官方判读后启动）

**触发条件**：fusion v3 @λ2.5 官方 overall ≥ -0.05（fusion 家族存活）→ 启动 v3.5；否则 v3.5 冻结，资源转 dualsrc 调优。

**设计**（tahoe_p12.parquet 已就绪：4,700 药扰组 × 33,092 基因域，300/300 靶响应基因命中）：
1. 辅助样本行：tahoe 药扰 (cell_line|drug) 组的 logfc 向量，特征 = basal(A549/SW480) + 模块 one-hot + **药扰特异的跨药签名统计**（无跨上下文 CRISPRi 签名可用，sig 特征位置置零 + has_sig=0 标志位），样本权重 0.3；
2. 主任务不变：6 上下文 CRISPRi 迁移（权重 1.0）——多任务共享骨干，期望药扰行教会 p53/IFN/EMT 模块的方向先验；
3. 训练顺序：先主任务 warm-up 5k steps → 加辅助行混训 → early stop on LOCO-HepG2。
**风险**：药扰行分布偏移可能稀释主任务（缓解：权重 0.3 + has_sig 标志位让模型可分）；**收益**：fid 模块缺口（p53/IFN）的唯一新信号源。

## fusion v3.5（Tahoe 药扰辅助多任务）实现与结果（2026-10-02 夜间，100min 时盒内完成）

实现按预案 2b79d02：`build_tasks.py --aux-tahoe`（aux 任务表 + basal reindex）、
`common.py`（build_sig_store allow_missing 空 store + sig_lookup 空守卫 + FeatureBuilder aux 支持：
sig 列显式置零 + **has_sig 第 22 特征列**）、train_xgb/train_mlp sample_weight（主 1.0/aux 0.3）
+ warm-up→混训（mlp 5k 步主任务预热 @bs65536；xgb 200 轮主任务预热续训）、模型对 **xgb_s35/mlp_s35**
（LOCO, s35）与 xgb_s2_35/mlp_s2_35_noemb（提交）。config `configs/l15_fusion/v35.yaml`。

- **aux 规模**：tahoe_p12 4,700 药扰组 → a549/sw480 两上下文各 94 组；任务表 93,966+93,726=**187,692 行**
  （总训练 30.78M 行的 0.61%），w=0.3；aux |logfc| 均值 0.155（主 0.076）；**has_sig 分布 1.0/0.0 二值，
  XGB gain share 2.3%**（模型确可用它分离 aux 行）；无 gwps 回退污染（sigstore_a549/sw480 不存在，
  sig_file 回退路径未触达 + allow_missing 守卫）。
- **CVCL 考证（资产纠错）**：basal_a549 ↔ CVCL_0023（A549, 全池 r=0.99998，4,056 DMSO cells 精确重建）；
  **"basal_sw480" 实为 CVCL_0546（SW620, 同患者转移株）r=0.99886**（CDX2 1.51 结肠✓；CVCL_0459 假设
  r=0.858 证伪）。上下文键沿用 sw480，map 已在 config 注明。另修复：build_aux 曾把 reindex 缓存写覆
  raw 资产（路径冲突）→ raw 重提取至 raw_basal_*.npy + 同路径禁止断言。
- **LOCO-HepG2 gate（λ2.5, anchor-normalized）**：

| 臂 | overall | pds | fid | nmae | reach | jac |
|---|---|---|---|---|---|---|
| v3（基线） | 0.3214 | 0.5741 | 0.8536 | 0.1440 | 0.2640 | 0.0925 |
| **v3.5 aux w=0.3** | **0.3313** | **0.6578** | 0.8562 | 0.1324 | 0.2486 | 0.0930 |

  **gate 通过（≥0.3114 线）且 +0.0099**：药扰辅助行没有稀释主任务，反而恢复 pds（v3 相对 v2 的
  −0.043 缺口被补回并超越 v2 的 0.6174）；未触发 0.15 降权重重试。mlp_s35 best_val 0.01478（v3 0.01434）。
- **A/B/C 提交物**：`outputs/submissions/20261003_l15fusion_v35.{h5ad,vcc}`（vcc prep 全绿，
  TMPDIR=/data/resource/wanglab/aivc/tmpdir）；validation 360,000×18,533，entries 2.085B，
  integer true。Δ̂ sanity：|Δ̂|>0.25 基因/靶 mean A/B/C = **11.7/8.1/6.8**（v3 8.5/4.4/3.8，
  v2 24.6/20.8/14.2——比 v3 略丰满，仍远紧于 v2），gwps 方向一致率 0.888-0.901。
- metrics：`outputs/metrics/20261003_l15fusion_v35.json`；LOCO 分数
  `outputs/metrics/local_bench/l15v35s1_fused_lam2.5/anchor_normalized.json`。
- **提交纪律**：v3.5 是否上报由 PI 按今晚 fusion v3 @λ2.5 官方分决定（≥−0.05 → v3.5 上；<−0.05 → 冻结）。
- 风险：① 药扰行剂量/池化噪声与 CRISPRi 因果方向不完全同构（w=0.3+has_sig 已对冲，LOCO 正增益实证）；
  ② aux 只覆盖 A549/SW480 两上皮系，对 context A(T 谱系)/C(IFN-high) 的迁移是间接的；
  ③ mlp warm-up 5k 步按 bs65536 实现（预案 5k@262k≈43 epoch 超总预算，已在 config 注明换算）。
