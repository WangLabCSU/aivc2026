# Bizard v0.3 结果汇总（2026-09-30/10-05 夜间窗口）

> 执行: ML 专家 ｜ 时间盒 150 min ｜ 代码: `src/models/bizard/v03_run.py` + `scripts/bizard_v03_{chain,bench}.sh`
> 配置: `configs/bizard/v03.yaml`（8 鸟四维表征变异）｜ 机读: `outputs/metrics/20261003_bizard_v03_exp.json`（实验）/ `20261003_bizard_v03.json`（提交物）
> 提交物: `outputs/submissions/20261003_bizard_v03.{h5ad,vcc}`（λ2.5, 全 8 鸟池 soft-rank）

## 1. 表征级变异实现（四维 × 8 鸟）

| 维度 | 鸟 | 实现 | 训练成本 |
|---|---|---|---|
| 骨干族 1 启发式 | heu_dual | dualsrc-deep（gwps+hct116 签名均值 logFC）, 零训练 | 0 |
| 骨干族 2/3 单臂 | xgb_s235 / mlp_s235 | v35 s2_35 XGB / MLP 单臂复用（α=1/0） | 0 |
| 骨干族 4 PBT 谱系 | v02_b3r | v0.2 b3_r（module_off+aux.45）复用 | 0 |
| 特征子集 bootstrap | xgb_sigdrop / mlp_sigsub | 互补镜像: 去 sig_{gwps,hct116} / 去 sig_{k562,rpe1,jurkat} | 2 训 |
| 训练行 bootstrap | xgb_rowboot | 70% 行采样(seed12) + reg:squarederror 双异质 | 1 训 |
| 损失族 | mlp_mse | MSELoss（v35 默认 Huber） | 1 训 |

工程: 启发式签名聚合缓存（每数据集一次 groupby 落盘 `heu_agg_*.parquet`, 与 load_delta **位级等价** maxdiff=0, 5.3→1.1 min/grid）; train_bird 支持 `mlp_loss`/`xgb_objective` 鸟级覆盖; 稠密网格严格对齐 soft-rank（启发式鸟 5 场景靶覆盖 100%, 零填充毒性在靶级不存在, 行级覆盖 91-97%）。

## 2. 鸟间相关矩阵（判据一: r≤0.9 → **部分达成**）

8 鸟两两 Pearson（Q_HepG2 为例, 全 5 场景见 `data/interim/bizard/v03/bird_correlation.json`）:
**heu_dual × 全部 fusion 鸟 r=0.58-0.68**（v0.1/v0.2 全对 0.9956 → 跨骨干彻底去相关）; fusion 族内部 0.93-0.99 仍高。
全对统计: **mean r=0.759**（v0.1 0.995）, max 0.996, 仅 45% 对 ≤0.9——严格逐对判据未过, 均值判据大幅达标。

**四维去相关功效排名**（跨 5 场景平均 r, 越低越好）:
1. **跨骨干（启发式 vs fusion）r=0.462** ← 决定性
2. 单臂拆分（XGB vs MLP）r=0.782
3. 特征子集（sig 源互补 dropout）r=0.847
4. 配置级（v0.2 module_off, 对照组）r=0.910
5. 行 bootstrap+squarederror r=0.975；损失族 MSE r=0.983（几乎无效）

## 3. 单鸟 Q_HepG2 bench 与聚合体

单鸟 Q: xgb_s235 **0.4146** / xgb_rowboot 0.4137 / v02_b3r 0.4082 / xgb_sigdrop 0.4012 / mlp_mse 0.3933 / mlp_s235 0.3932 / mlp_sigsub 0.3833 / **heu_dual 0.3031**（λ3.5, 家族口径）——表征变异以 0.01-0.03 的个体代价换取去相关, 启发式鸟弱 0.11。

聚合体（全 8 鸟等权池, soft-rank, λ2.5）五场景: **0.2449** {P .0289/Q .3710/R .3704/S_RPE1 .1342/**S_HCT .3199**},
对照 v0.1 0.2550 {P .0689/Q .4110/R .3841/S_RPE1 .2085/S_HCT .2025} → **−0.010 未过判据**, 但结构剧变:
**S_HCT116 +0.117**（启发式签名对 hct116 靶覆盖 100% vs gwps 54% 的场景, 混骨干互补兑现）, Q/R −0.01~−0.04,
P_H1/S_RPE1 被弱 heu 鸟拖累（P_H1 fid −0.06/reach −0.05）。聚合 Δ̂ vs 鸟均 r=0.75（v0.1 ≈1.0）——聚合体首次"不同于任何单鸟"。
**加权池**（heu w=0.25）Q **0.3897** vs 等权池 0.3710（+0.019, 仍低于 v0.1 Q −0.021）——权重是正确的修复方向但 0.25 仍偏重。

## 4. 判据与结论

- 判据（聚合 ≥0.2550 且逐对 r≤0.9）: **未过**（0.2449, −0.010; 逐对 max 0.996）。但两半判据各自取得实质进展:
  均值 r 0.759（v0.1 0.995）+ S_HCT116 +0.117 证明表征变异的方向收益真实存在; 失分集中于弱鸟等权入池。
- 混骨干方向成立: 跨骨干 r=0.46 是两代配置级变异（r≈0.995）从未接近的去相关量级; 代价是弱鸟入池毒性 → v0.4 用**加权 soft-rank**（w∝单鸟适应度, 或弱鸟 w≤0.1）而非等权。

## 5. v0.4 建议

1. 加权聚合定标: Q 场景扫 w_heu ∈ {0, .05, .1, .25}（工具已备 `aggregate(weights=)`）, 预期 w≤0.1 时 Q 回到 ≥0.405 且保留跨骨干去相关; S_HCT116 型场景可场景化加权（按单鸟该场景分）。
2. 骨干族再扩: tahoe-only 药扰臂 / dualsrc-confv2 / L1 conf 权重表作为第 5-6 族; 弱鸟只供去相关不供均值。
3. 关闭行 bootstrap/MSE 路线（r>0.97 无去相关价值）; sig 互补子集保留（r=0.85 且个体仅 −0.01）。
4. PBT 第 3 代在跨骨干杂交代上跑（heu×fusion 特征拼接或堆叠蒸馏）, 替代同族自交。
5. 提交策略: 10/6 窗口主候选仍为 bizard v0.1（Q 0.4110）; v0.3 vcc 作为多样性候选存档, 待加权定标后再升格。

## 6. 交付物与偏差

- 提交物: `outputs/submissions/20261003_bizard_v03.h5ad`（4.3GB, 360000×18533, integer）+ `.vcc`（vcc prep 已跑, 见 §3 日志）
- 机读: bird_correlation.json / dim_decorrelation.json / bench tags `bz03_*`（8 鸟 Q 全覆盖 + 聚合 5 场景）
- 偏差: ① 训练鸟单分支（XGB 训 XGB 臂/MLP 训 MLP 臂, α 对齐 1.0/0.0）——v0.2 全融合口径的偏差, 换取臂级变异; ② 聚合 bench 中途 worker 被杀, 断点续评补齐; ③ 时间盒内未跑 λ3.5 聚合口径与加权池五场景 bench。
