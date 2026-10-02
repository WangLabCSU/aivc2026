# 迭代 007：双源平手、ensemble 计数级证伪、多锚点普适（专家 D）

## E1 双源启发式（gwps+hct116 @λ3.5）
Q_HepG2 归一化 0.2901 vs 单源 0.2871（+0.003 平手）。hct116 增益落在 **nmae +0.059**（真实），fid 仅 +0.004（假设证伪），代价 pds -0.023。定性："不伤"而非"提升"。三源=dual（norman 对 Q 零贡献）。

## E2 ensemble
- 计数级两变体证伪：half-up 0.2294（质量+9.4%膨胀）、banker's 0.0780（nmae -0.74）——根因：计数空间算术平均≥几何平均（Jensen gap），方向相反基因产生虚假 LFC。
- Δ̂ 级补做（fusion 对 Q 靶 pre-λ Δ̂ 已存 pred_s1bench_hepg2_v2_fused.parquet，回归 slope≈2.45 corr>0.99）：**0.3200**（fid 0.807↑、pds 0.729 部分保留）——但含 50% fusion Δ̂，家族官方迁移失败实测（+0.055→-0.057），不冒进。

## E3 多锚点复制（普适性验证）
P_H1 ±0.0142、R_Jurkat ±0.0210、Q ±0.0282——三上下文 ref 选择噪声均为 MC 细胞噪声 10×+；<0.04 平手规则全上下文安全；单 ref 有偏（R seed-0 低估 3-ref 均值 -0.037）。公式自检发现官方 mse clamp_low=0（P_H1 线性 -0.127→scored 0）。

## 明日两发（已构建 prep 全绿）
- daily-best：**dualsrc λ3.5**（20261003_dualsrc_lam35.vcc）——官方预期 ≈ +0.025~0.03
- calibration：**dualsrc λ4.5**（20261003_dualsrc_lam45.vcc）——λ 阶梯封顶检验（本地 nmae 已崩，仅当官方斜率线性才 +0.011）
