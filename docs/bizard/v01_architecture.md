# Bizard-v0.1 架构设计（深度学习 + 强化学习 + 鸟群优化）

> 2026-10-08 ｜ 目标：排行榜前 3（当前 Rank 633，+0.0886 → 目标 +0.3~0.5）
> 核心：上下文条件化深度模型 + 强化学习奖励优化 + 鸟群多场景局部最优聚合

## 1. 为什么当前架构到顶了

**根因诊断**（10 发官方数据 + 五场景 bench 闭环）：
- Δ̂ = 靶基因伪批量 logFC 均值 × λ 缩放——**上下文无关的线性变换**
- K562/HCT116 源缺失的模块（p53 轴/IFN/谱系 TF）方向在 A/B/C 上不可学——fid 天花板 -0.034
- bench 的 fid 盲区（A/B/C 特异坍塌不可复现）——本地无法迭代 fid

**突破方向**：从"平均响应缩放"到"上下文条件化响应预测"——模型必须学会**同一个靶基因在不同上下文中产生不同的方向模式**。

## 2. Bizard-v0.1 架构

### 2.1 深度学习骨干：Context-Conditioned Response Network (CCRN)

```
输入：
  context_emb(ctx) — 上下文嵌入（NTC 伪批量 → 降维，或 one-hot + module scores）
  target_emb(g) — 靶基因嵌入（基因身份 +跨源签名统计）
  gene_emb(j) — 响应基因嵌入

网络：MLP (2048 → 1024 → 18533) 或 Attention-based
  input = concat(context_emb, target_emb, gene_emb, cross_features)
  output = Δ̂(j | g, ctx) — 全基因 logFC 预测

训练：6 CRISPRi 上下文全量监督（30M+ 样本行）
  loss = Huber(Δ̂, y_true) + 方向一致率正则
```

### 2.2 强化学习层：官方分数驱动的策略优化

```
状态 s = (当前模型参数, bench 五场景分数)
动作 a = (λ, per-source weights, per-module confidence, carrier 参数)
奖励 r = 官方 overall（每 2 发/日）

策略：PBT（Population-Based Training）
  - N=8 个 worker，每个持有不同的 (模型权重, 超参) 状态
  - 每 eval round：所有 worker 提交 → 读取官方分 → 按适应度排序
  - exploit：末位 worker 复制头位权重
  - explore：复制体加高斯扰动
  - 迭代：日复一日，2 发/日 的官方信号持续引导收敛
```

### 2.3 鸟群聚合：多场景局部最优 → 全局

```
N 个训练好的鸟（不同场景偏置 + 不同架构家族）
→ 五场景 bench 评估 → soft-rank 聚合（场景化加权）
→ 全局最优 pred = 跨家族互补融合
```

## 3. 与现有系统的关系

| 组件 | 来源 |
|------|------|
| 训练数据 | 6 CRISPRi 上下文（gwps 深 17k/hct116 深 17k/rpe1/jurkat/hepg2/k562_ess）+ Tahoe aux |
| 载体 | NTC bootstrap 采样 ⊕ 2^(λ·Δ̂)（已验证） |
| bench | 五场景 anchor 归一化（已验证 + affine 校准） |
| 骨干 | fusion v3.5 管线（已验证）+ bizard 表征级变异（已验证 r 0.759） |
| 新增 | CCRN 深度模型（上下文条件化）+ PBT 强化学习层 + 鸟群聚合器 |

## 4. 路线图

| 版本 | 内容 | 目标 |
|------|------|------|
| bizard-v0.1 | CCRN 骨干 + 6 上下文全量训练 + 5 场景 bench | 官方 ≥ +0.10 |
| bizard-v0.2 | + PBT 强化学习层（官方分驱动策略优化） | fid ≥ -0.15 |
| bizard-v0.3 | + 鸟群聚合（跨家族多场景局部最优） | overall ≥ +0.15, Rank ≤ 500 |
| bizard-v1.0 | + State 符号 ensemble + scBaseCount 增源 | **Rank ≤ 100** |
