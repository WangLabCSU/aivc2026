# 迭代 012：Bizard CCRN 上下文条件化响应骨干（深度学习专家，~60 分钟）

## 交付清单

- CCRN v0.1：38 维输入 → MLP(2048→1024→1)，2,179,073 参数，Huber+AdamW+cosine，GPU 常驻 28.4M 行（14s/epoch，LOCO 臂 98s / 全量臂 159s）
- LOCO-HepG2 @λ2.5（anchor 归一化）：**CCRN 0.3474** vs fusion v3.5 s1 0.3313（+4.8%）vs deep-only 消融 0.3240；multi-ref 0.3412±0.0224
- A/B/C 提交物（λ3.5）：`outputs/submissions/20261008_bizard_ccrn_v01.{h5ad,vcc}`，prep 全绿（360k cells × 18,533 genes，官方靶校验通过，密度余量 56.2%）
- 代码：src/models/bizard/ccrn.py + configs/bizard/ccrn.yaml

## 架构与归因

输入 = fusion v3.5 的 22 维逐行特征 ⊕ 上下文指纹 9 维（6 模块 basal 分数 + basal 均值/SD/非零占比，全部可由 NTC basal 推出 → 对 LOCO 与未见上下文可泛化）⊕ 上下文模块分数×基因模块标志交互 6 维 ⊕ basal 中心化 1 维。三臂对照（Q_HepG2 anchor）：deep-only(22 维) 0.3240 < fusion v3.5(浅 MLP) 0.3313 < CCRN 0.3474——**增益来自上下文条件化而非深度**；逐指标 vs fusion：reach +0.073、nmae +0.068、fid +0.012，pds −0.059。+0.016 在 multi-ref 噪声带（±0.022）内，判读为"持平略优"。

## 关键工程教训

1. A/B/C 面板缺 RPS/RPL → ribo 指纹=0（−7σ OOD）→ 模型外推漂移（Δ̂ 均值 +0.26）；NaN+训练中位数插补修复后，Δ̂ 与 fusion 的 target 内中心化相关从 ≈0 升至 0.60-0.62（互补且同向）。
2. 未见上下文 Δ̂ 仍有均值漂移（−0.12/−0.25/−0.40）→ 提交配方加 per-context 均值中心化（对齐家族约定 ≈0）。

## v0.2 建议（PBT 接入点）

- CCRN 训练仅 ~2 分钟/代 → PBT worker 状态=（λ、指纹维度缩放、aux 权重、hidden 宽度、seed），exploit/explore 可高频迭代；奖励=LOCO bench（本地）+官方 2 发/日
- λ 曲线需按 CCRN 家族重标定（bench λ2.5→6.5 扫描；中心化后谱间 sd 0.06-0.125 略宽于 fusion，λ 敏感性待测）
- 风险纪律：bizard 家族 bench→官方历史上不迁移（v02 PBT −0.0200），按 local−0.42±0.1 管理预期；本提交 λ3.5 为保守首发

## 产物

configs/bizard/ccrn.yaml；src/models/bizard/ccrn.py；data/interim/bizard/ccrn/（模型/损失曲线/Δ̂）；outputs/metrics/local_bench/bzccrn1_loco_lam2.5/、bzccrn1ab_deep22_lam2.5/anchor_normalized.json；outputs/metrics/20261008_bizard_ccrn_v01.json
