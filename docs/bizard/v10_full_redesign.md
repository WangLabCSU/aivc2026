# Bizard-v1.0：端到端深度学习 + 强化学习 完整重设计

> 2026-10-08 ｜ 战略转向：放弃增量微调路线，从零构建端到端深度学习架构
> 目标：官方 overall ≥ +0.15（当前 +0.0886），fid ≥ -0.10，Rank ≤ 400
> 竞赛截止：11/5（剩余 30 天，约 60 发官方额度）

## 1. 战略诊断：为什么增量路线到顶了

**根因**：所有已提交方案（10 发）本质上都是「伪批量 logFC 均值 × λ 缩放」——上下文无关的线性变换。改进空间被三个天花板锁死：
1. λ 阶梯封顶（λ6.5 = +0.0886，λ7.5 本地崩塌）
2. fid 三条修复路线证伪（幅度缩水/top-K/方向先验子集）
3. fusion/bizard 家族 bench 增益不迁移官方（offset -0.42）

**突破需要**：从「手工特征 + 线性缩放」到「端到端深度学习 + 强化学习」——模型自行从数据中学习上下文条件化的响应模式。

## 2. 数据整合（全量）

| 数据 | 规模 | 用途 |
|------|------|------|
| 6 CRISPRi 上下文 | 30.78M 行 (靶g × 响应基因j × logfc) | 主监督 |
| HCT116 深度（109 批 3.4M cells） | 17,358 perts，282/300 竞赛靶 | 主监督 + fid 修复源 |
| Tahoe plate12 药扰 | 4,700 组，300/300 靶命中 | 辅助任务 (w=0.3) |
| Orion HCT116 全量 | 3.4M cells，17,358 perts | 主监督（HCT116 深度版） |
| A/B/C NTC basal | 3 × 18,400 cells | 推理载体 + 上下文条件 |
| scBaseCount | 按需 | 增源（v1.1） |

## 3. 模型架构（Bizard-v1.0）

### 3.1 BizardNet：Context-Conditioned Perturbation Response Transformer

```
输入 token：
  [CTX] 上下文 token → NTC 伪批量 (18533 dim) → linear proj (512)
  [TRG] 靶基因 token → 基因嵌入 (从预训练 gene encoder 或 learned)
  [GEN] 响应基因 token × N → 基因嵌入 + 位置编码

Transformer Encoder：
  layers = 6, d_model = 512, n_heads = 8, d_ff = 2048
  input = [CTX_proj, TRG_proj, GEN_proj × N]

输出头：
  Δ̂ head: linear(512 → 18533) — 全基因 logFC 预测
  fid head: linear(512 → 1) — fid 预测（辅助任务，帮助学方向）
```

### 3.2 训练策略

- **Stage 1**：6 上下文全量监督（30M 行），batch = (ctx, target, response_gene) 三元组
- **Stage 2**：+ Tahoe 药扰辅助（w=0.3，has_sig 标志位）
- **Stage 3**：LOCO 微调（hold out HepG2 验证迁移）
- Loss: Huber(Δ̂, y_true) + BCE(fid_pred, fid_true) + 方向一致率正则

### 3.3 强化学习层（PBT）

```
策略空间：学习率 / dropout / layer 数 / d_model / aux 权重 / λ / carrier 参数
奖励：五场景 bench anchor overall（affine 校准 → 官方预测）
算法：PBT（exploit + explore）每轮淘汰末位、复制头位
迭代：每代 ~40 分钟，3-5 代/天
```

## 4. 评估与提交

- 本地：五场景 bench（Q_HepG2/R 优先，LOCO 严格）
- 官方：每日 2 发，择最优
- 里程碑：bizard-v0.1 ≥ +0.10 → v0.2 + PBT → ≥ +0.15 → v0.3 + 鸟群 → 冲前 3
