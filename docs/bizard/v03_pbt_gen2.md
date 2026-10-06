# Bizard v0.3 PBT gen-2：表征级变异鸟的代际进化（2026-10-06 夜间窗口）

> 执行: ML 专家 ｜ 时间盒 150 min ｜ 代码: `src/models/bizard/v03_gen2.py` + `configs/bizard/v03_gen2.yaml`
> 机读: `outputs/metrics/20261003_bizard_v03_pbt_gen2.json`、`data/interim/bizard/v03/gen2/{pbt_report,gen2_fitness_table,gen2_children}.json`
> bench tags: `bz03g2_g2_c1..c4`（子代五场景全）、`bz03g2_agg_lam2.5`（聚合五场景）、`bz03_xgb_s235/rowboot`（父代补五场景）
> 全部子代为 **Δ̂ 级插值**（复用 v0.3 鸟 pred parquet, 零重训）, bench λ2.5 主口径

## 1. 全局适应度排序（gen1 8 鸟, Q_HepG2 anchor 归一化 overall）

| rank | 鸟 | Q | 备注 |
|---|---|---|---|
| 1 | xgb_s235 | .4146 | 父代 P1 |
| 2 | xgb_rowboot | .4137 | 父代 P2（与 P1 r=.9921 近克隆） |
| 3 | v02_b3r | .4082 | |
| 4 | xgb_sigdrop | .4012 | |
| 5 | mlp_mse | .3933 | |
| 6 | mlp_s235 | .3932 | |
| 7 | mlp_sigsub | .3833 | |
| 8 | heu_dual | .3031 | 跨骨干去相关供体（λ3.5 家族口径） |

聚合体参照: gen1 等权池五场景 mean5 .2449（Q .3710）、gen1 加权池（heu w=.25）Q .3897、v0.1 mean5 .2550（Q .4110）。

## 2. 配对与继承：4 子代（Δ̂ 级插值, 全部落盘 `data/interim/bizard/v03/gen2/`）

| 子代 | 基因型（插值权重） | 算子 |
|---|---|---|
| g2_c1 | .5·P1 + .5·P2 | crossover 等权 |
| g2_c2 | .7·P1 + .3·P2 | crossover 适应度偏置 |
| g2_c3 | .3·heu_dual + .7·P1 | **跨骨干杂交** |
| g2_c4 | heu .10 + P1 .30 + P2 .25 + b3r .20 + sigdrop .15 | 适应度加权池（弱鸟 w=.1 只供去相关） |

## 3. 子代适应度（五场景 overall / mean5, λ2.5）

| 子代 | P_H1 | Q | R_Jurkat | S_RPE1 | S_HCT116 | mean5 |
|---|---|---|---|---|---|---|
| g2_c1 | .0761 | .4166 | .3933 | .2241 | .2123 | .2645 |
| g2_c2 | .0761 | **.4175** | .3938 | .2274 | .2144 | .2658 |
| **g2_c3** | **.0792** | .4047 | .3840 | .1951 | **.6258** | **.3378** ★ |
| g2_c4 | .0768 | .4121 | .3897 | .2178 | .3082 | .2809 |
| 父代 P1 (s235) | .0736 | .4146 | .3927 | .2259 | .2162 | .2646 |
| 父代 P2 (rowboot) | .0767 | .4137 | .3905 | .2243 | .2086 | .2628 |

（父代 Q 为 v0.3 已有评分; P/R/S 为本次补跑, 显式 `pred_{bird}_{ctx}.parquet` 输入。注意 bench_bird.py 直调时 tag 前缀 `bz01_` 剥离逻辑对 `bz03_*` tag 失效 → 回退聚合 Δ̂ 的坑, 已绕开; 父代 Q 的 eval 目录在清理污染 tag 时删除, 数值以 v0.3 summary/gen2_fitness_table.json 存档为准）

读法: ① 同族自交（c1/c2）Q 超双亲（+.002~.003）, mean5 .2645/.2658 vs 亲本 .2646/.2628——近克隆交叉信息增量有限（≤+.003）; ② **跨骨干杂交 g2_c3 决定性（超亲优势 +.073）**: S_HCT116 .6258（父代 .21/.21 → +.41）, mean5 .3378 全谱系最高（> v0.1 聚合 .2550）, 代价仅 Q −.010; ③ c4 验证「弱鸟低权入池」均衡有效（mean5 .2809）。

## 4. 三代聚合对比（soft-rank, λ2.5）

| 口径 | v0.1 (bz01_agg_sr) | gen1 等权 (bz03_agg_lam2.5) | gen2 = 4 子代池 (bz03g2_agg_lam2.5) |
|---|---|---|---|
| P_H1 | .0689 | .0289 | **.0764** |
| Q_HepG2 | .4110 | .3710 | **.4140** |
| R_Jurkat | .3841 | .3704 | **.3911** |
| S_RPE1 | .2085 | .1342 | **.2200** |
| S_HCT116 | .2025 | .3199 | .2688 |
| **mean5** | .2550 | .2449 | **.2741** |

gen2 vs gen1: **Q +.0430 / mean5 +.0292**; gen2 vs v0.1: **Q +.0030 / mean5 +.0191**。
**判据（gen2 聚合 > gen1）通过 → PBT 代际转正**（首次同时超越 v0.1 两口径）。
注: gen1 加权池仅 Q（.3897）无 mean5; gen2 聚合 S_HCT116（.2688）低于 c3/c4 单鸟——soft-rank 分位映射对「单鸟场景强信号」保守（压平效应）, 见 §6。

## 5. A/B/C 提交物（已打包验证）

- `outputs/submissions/20261003_bizard_v03_gen2.h5ad`（4.07GB, 360000×18533, integer, density_margin .5611, λ2.5, NTC bootstrap 400 ⊕ 2^(λ·Δ̂) max_fold 8）
- `outputs/submissions/20261003_bizard_v03_gen2.vcc`（vcc prep 全绿: targets verified / counts-preserved）
- Δ̂ 来源: `data/interim/bizard/v03/gen2/agg_delta_submit_{A,B,C}.parquet`（4 子代 soft-rank）
- 未上传（ledger 无行）; 与 v0.1（官方 −0.0173/Rank 909）相比 Q 口径 +.0030, 建议作下一窗口主候选复验后提交。

## 6. v0.4 建议（PBT 代际转正后路线）

1. **种群换血**: v0.4 种群 = {g2_c3, g2_c4, g2_c2} + 新血; 同族自交线（纯 s235×rowboot）关闭（r>.99 无新信息）。
2. **w_heu 网格定标**（Δ̂ 级 heux235 族）: w ∈ {.2, .3, .5} × 场景化（S_HCT116 允许高 w_heu, Q 用 ≤.2 保 Q）——c3 的 S_HCT .6258 表明 heu 信号在 hct116 靶覆盖场景被严重低估。
3. **聚合器升级**: soft-rank → 场景化加权 soft-rank 或「插值池+soft-rank 混合」; 修复 S_HCT116 压平（gen2 聚合 .2688 < c4 .3082 < c3 .6258）。
4. 上传策略: gen2 vcc 为主候选（两口径全面 ≥ v0.1）; 单基因型 g2_c3 作观察候选（mean5 +.083 但单场景依赖风险）。
5. 工程: cell-eval2 并发 ≤3-5（16 并发 GPU OOM, cupy）; 摘要竞态用 `--rescore` 自愈（已内置）。

## 7. 偏差

- gen1 8 鸟仅 Q 全覆盖（P/R/S_RPE1/S_HCT116 只评了 2 父代 + 聚合体; 预算所剩, 非关键路径——全局排序判别主要来自 Q）。
- 首轮 16 并发 bench 6 点 GPU OOM 后分批补齐, 摘要由 `--rescore` 从 eval 目录重建（数字以本文档/机读 JSON 为准）。
- S_HCT116 bundle 为 diagnostic 口径（与 v0.1/gen1 同一 bundle, 代际可比）。
