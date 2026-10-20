# bizard 大架构设计草案 (v0.11 方向) — 2026-10-20

## 已核实的数据事实 (tmp/bzarch_inspect*.py)
1. vcc2025 官方公开数据 = Flex_1/2/3 三组 batch (每组16批), **同 151 targets 含 NTC** 全空间重复
   → Flex_1/2/3 就是"三个 context"的类似物! (221273 cells × 18080 genes)
2. 当前挑战 = contexts A/B/C × 300 targets × 18080 panel (controls manifest vcc2026-val-1)
3. **vcc2025 targets 与当前 300 targets 重叠仅 25/300** → 学的是"机制迁移"不是"靶点记忆"
4. vcc2025 Validation (50 targets, 98927 cells) = 官方给的现成 held-out-pert 验证集!
5. 官方 gene panel 重叠: 18076/18533

## 结构同构性 (核心洞察)
当前挑战结构: NTC(context) ⊕ Δ(target, context) → 138400 cells/pert-context
vcc2025 结构: NTC(Flex组) + 扰动 cells, 每 Flex 组全 151 targets 重复
→ 同构任务: 给定 (target, source-context) 的响应, 预测 (target, unseen-context) 响应
→ **训练数据现成: Flex_1/2 → Flex_3 跨 context 迁移, held-out = Flex_3 的真值**

## 大架构方案: Embedding-Conditioned Response Decoder (ECRD)
输入 (每 (target, gene)): 
- SE-600M protein embedding 5120-dim × 2 (target基因 + 被预测基因) → 投影到 d
- NTC context profile (mean log-normalized, 18080→d via 线性+ pool)
- per-gene base stats (mean expression, variance)
输出: Δ(target, context) per gene — 学的是 "扰动响应如何随 context 变化"
损失: 对 vcc2025 logFC (per Flex 组 pseudo-bulk vs NTC同组) MSE
验证: LOCO-Flex (Flex_1/2 训练 → Flex_3 预测), 对比基线 = 跨 Flex 组平均签名 (≈现有做法)
成功判据: held-out Flex_3 logFC 预测 spearman/nmae 显著优于"跨组均值"基线
集成: 预测的 Δ 替换/混合现有 Δ 长表 → 走 HY8 pipeline → bench 链全链路

## 训练成本估计
样本 = (target 151 × gene 18080) × (train Flex 组 2) ≈ 5.5M rows, 特征 ~10k+5120×2 投影
→ 2 层 MLP (d=1024) + gene-target 双塔 → 单 GPU 数小时内可训
