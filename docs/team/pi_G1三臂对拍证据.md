# PI 汇总：G1 三臂对拍证据包（2026-10-14 ~05:20 EDT）

用途：R-1 预注册 G1–G4 裁决的完整证据。数据全部来自
`outputs/metrics/local_bench/{tag}/Q_HepG2_scored.csv`（from_replicate 列 = 官方口径近似）与
`data/interim/xgb_amp/bench_arms/*_Q_HepG2.parquet`。7 个未解析靶（CYS1/HIST1H2BM/LPA/POU5F1B 等）
未剔除，对三臂同等生效，组间比较有效。

## 四臂总表（from_replicate，overall = 6 指标均值）

| 臂 | overall | Δ vs 基线 | nmae | mse | fid_yield | reach | jac | pds |
|---|---|---|---|---|---|---|---|---|
| 基线 bzbiz10skip_lam16 | 0.35209 | — | 0.19954 | 0.0 | 0.86467 | 0.38661 | 0.09794 | 0.56376 |
| xgbamp_rec（重定心 g） | **0.36179** | **+0.0097** | **0.29553 (+48.1%)** | 0.0 | 0.84417 (−0.0205) | 0.35725 (−0.0294) | 0.09678 (−0.0012) | 0.57702 (+0.0133) |
| xgbamp_raw（原始 g） | 0.35020 | −0.0019 | 0.11305 (−0.0865) | 0.0 | 0.86009 | 0.42393 (+0.0373) | 0.10058 | 0.60355 (+0.0398) |
| oracle3（oracle g） | **−0.31092** | −0.6630 | **−3.92291** | 0.0 | 0.84647 | 0.49255 (+0.1059) | 0.09838 | 0.61999 (+0.0562) |

oracle3 的 from_baseline nmae = −0.6817（同方向，幅度较小）。

## g/delta 分布（bench_arms parquet，96 靶 × 10020 基因）

- oracle3 delta: mean −0.0077, std **0.0362**, range [−1.295, +1.068]
- rec delta: mean −0.0027, std **0.0111**, range [−0.393, +0.227]（≈ oracle 宽度的 1/3.3）

## PI 解读（供 REV 裁决，不构成结论）

1. **有价值的信号 = 重定心后的逐靶相对结构（E[g]≈1），不是绝对幅度。**
   绝对幅度即使 oracle 级完美也灾难性摧毁 nmae（−3.92 from_replicate），仅方向类
   指标受益（reach +0.106 / pds +0.056）。raw 臂（绝对但模型学）同样 nmae 恶化
   （−0.0865），方向指标改善。rec 臂反转：nmae 大幅改善、方向指标小幅回吐。
2. **G1 判据字面**：P_H1 = "nmae 聚合增益 ≥5% 或 mse 可复现改善"。
   rec 臂 nmae 组件 +48.1% ≫ 5% → 字面 PASS；mse 全臂恒 0 → mse 分支不可用
   （本地幅度缩放在 λ16 无法把 mse 推离 0，连 oracle 也不行——mse≠0 探针的
   本地预注册价值已死，官方是否不同未知）。
3. **裁决点（需 REV 认定）**：预注册写 "G1 oracle-g 对拍"。三臂设计中 oracle 臂
   灾难失败、模型 rec 臵通过。判据绑定的对象是「oracle 上限证明 g 结构有效」
   （该目的由 rec vs raw vs oracle 三臂对照完成：相对结构有效、绝对结构有害）
   还是「oracle 臂本身须达标」（字面更严，已 FAIL）。R-1 只许收紧不许放松，
   PI 不自行放宽，请 REV 依预注册原文裁定。
4. **G3 现状（独立于 G1）**：loco_frozen.json pass_all_ctx=false，
   rpe1/jurkat/hct116 三轴 rmse_model > rmse_g1（0.878/0.495/0.650 vs
   0.760/0.415/0.568），无多重 split 方差数据 → 豁免条件不满足 → 倾向 FAIL。
   若 G3 FAIL → 不推荐发档 → slot1 默认 λ47 → 官方名 bizard-v0.3。
5. G2：推理饱和 0/0（bench+submit 全表），train-implied 1.52%/4.87% ≤15% → 预期 PASS。
   G4：mse 声明不涉及 affine 外推，n/a。

---

## 附录：对读 gate_report 后的口径和解（2026-10-14 06:05 EDT 增补）

收到 ml-expert `outputs/xgb_amp/gate_report.md`（G1 FAIL · G2 PASS · G3 FAIL · G4 遵守）。
与本文件四臂总表的数字差异源于**两套口径**，不是数据矛盾：

1. **口径 A（本文件总表）**：`Q_HepG2_scored.csv` 的 from_replicate 列 = 官方评分仿真口径；
   mse 恒 0（本地工件，幅度缩放推不动），nmae 为 score 型（oracle3 −3.92 可为负）。
2. **口径 B（gate_report 表）**：RAW 直算口径（误差型 nmae 0.88 级、mse 0.019 级）。
   **R-1 G1 契约明写"口径复用探针2"→ 绑定口径 B**。
3. 方向一致：两口径都显示 rec 臂 nmae 改善（A: score +48.1% ↔ B: 误差 −1.5%），
   但 B 口径下增益 1.5% < 5% 且在种子噪声带（Δ<~0.04）内 → **G1 FAIL 成立，PI 接受**。
   本文件第 2 条"rec 臂字面 PASS"的解读作废（误用了非契约口径）。
4. **"oracle ≡ uniform ×3" 与本文件 delta 分布自洽**：oracle g 96/96 全部 clip 饱和至 3.0，
   幅度线性 ×3（λ16→λ48）；parquet delta std 0.0362 ≈ rec std 0.0111 × 3.26，
   正是均匀 ×3 缩放的期望——本文件此前"oracle delta 非均匀、质疑饱和说"的推测**撤回**。
5. mse 差异同源：口径 A mse 恒 0 vs 口径 B mse 有变化（oracle +0.0631 恶化），
   与 mse=0 streak 的既有认知一致。
6. **PI 最终结论**：G3 FAIL（rpe1/jurkat/hct116 相对劣化 14.6–19.1%，无多重 split 方差，
   豁免不启用）**独立构成否决**——无论 G1 口径之争如何裁定，XGB-MVP 均不推荐发档。
   slot1 按预注册默认发 λ47；按用户序列编号规则（版本号跟随提交序列），λ47 作为当日
   第 1 发 → 官方名 **bizard-v0.2**（此映射覆盖 a1312c4 的"λ47=slot2/v0.3"内容绑定）。
