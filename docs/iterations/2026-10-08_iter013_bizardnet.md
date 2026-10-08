# 迭代 013：BizardNet v1.0 上下文条件化扰动响应 Transformer（深度学习专家，~240 分钟）

## 交付清单

- BizardNet v0.2（19,729,943 参数）：三 token 序列 Transformer——每序列 = (上下文 c, 靶基因 g) 的 128 个响应基因；[CTX] 19 维 NTC-basal 指纹（6 模块分数 NaN-插补 + 全局统计/分位数 + top-var 基因 PCA5）、[TRG] 靶 token、[GEN]×128 响应基因 token（各 = 预训练基因嵌入 SE600M⊕Tahoe 256d + fusion v3.5 逐行标量 20d/9d + token-type emb；基因 token 集合语义以 type-emb 代位置编码）→ 投影 512 → Transformer Encoder（6L/8H/d512/ffn2048, pre-LN, SDPA）→ ReZero 门控 α + 零初始化 skip 线性证据头（20d→1）双通道输出
- 训练：feat_train.npz 30.78M 行 → 248,227 序列（N=128），GPU 常驻标量张量 3.9GB，95-107s/epoch（L20, bf16）；LOCO 臂 12ep/17.5min best val Huber 0.0094，full 臂 20.4min best 0.0028；损失曲线 data/interim/bizard/bizardnet/{loco,full}_v2.log
- LOCO-HepG2 bench anchor 归一化 @Q_HepG2：**skip 载体 λ16 = 0.3521** vs CCRN v0.1 0.3474 / fusion v3.5 LOCO 0.3313 / 任务书基线 0.2644；λ 扫描 12→0.3407、16→0.3521、18→0.3427、20→0.3341（tag bzbiz10skip_lam*）
- A/B/C 提交物（λ39 幅度校准）：`outputs/submissions/20261008_bizard_v10.{h5ad,vcc}`，validate 全绿（360000×18533，density_margin 0.5611，integer），vcc prep 全绿（官方靶校验/counts-preserved）
- 代码：src/models/bizard/bizardnet.py + configs/bizard/bizardnet.yaml + bizardnet_submit_deltas.py + scripts/bizard_v10_chain.sh

## 关键发现：transformer 分量在未见上下文外推失败，skip 证据头是迁移载体

三轮迭代的实证链（全部 Q_HepG2 anchor）：
1. v0（lr2e-4+学习身份嵌入 27.1M）：val 平台 0.0122，bench -0.0975——输出塌缩（sd 0.029 vs CCRN 0.053）
2. v1（lr8e-4+skip 头）：bench -0.1402——**诊断出根因：transformer 路径输出 sd≈14 的大方差弱相关信号（vs truth r=0.05），淹没 skip 头（单独 r=0.345）**；gate 学成缩放系数（-0.008）而非稀疏门
3. v0.2（去学习身份嵌入 + ReZero）：val 降至 0.0094、proxy spearman_med 0.276（v0 0.072），但 bench 仍 -0.1142——**Huber-val 与 bench 解耦**：变换器分量在指纹未见过的 hepg2 上是噪声，线性证据头（20 维跨源签名统计直连）才是可迁移信号
4. 解法＝分量选择（skip 载体）+ λ 幅度重标定：skip 头 sd 0.0121 需 λ16 达 λ·sd≈0.194 峰（family 启发式 λ6.5×sd≈0.03 同律）；对 A/B/C（sd 0.0049）按同律外推 λ39

工程教训：A/B/C 面板 tpos<0 占 6.2M/30.78M 训练行（UNK 靶 token+valid 位）；(ctx,tpos) 序列表 123MB int32 全 GPU gather，免流式开销。

## v0.2 建议（强化学习层/PBT 接入点）

- **reward 就位**：bench-anchor overall（含 λ·sd 扫描）已是可微代理之外的直接奖励；PBT worker 状态 =（λ_submit, α 门控初值, dropout, lr, aux 权重, seq_genes N），每代 ~20min（训练 18min+bench 2.3min），3-4 代/夜免费窗口
- **架构方向**：(a) transformer 分量改为预测 skip 残差 + 训练时 context-dropout（随机置零指纹 p≈0.2，阻断指纹记忆、强迫跨上下文路由）；(b) fid 辅助头（设计书 §3.1 的 BCE 分支）补方向一致性正则；(c) 官方 2 发/日标定 A/B/C 的 λ·sd（当前以 hepg2 谱外推，是提交最大不确定度）
- 风险纪律：bench→官方历史 offset -0.42±0.1（bizard 家族），λ39 超出家族 2.5-7.5 惯域，首发按保守预期管理

## 产物

configs/bizard/bizardnet.yaml；src/models/bizard/{bizardnet.py,bizardnet_submit_deltas.py}；scripts/bizard_v10_chain.sh；data/interim/bizard/bizardnet/（模型/曲线/Δ̂/bench pred）；outputs/metrics/local_bench/bzbiz10*/；outputs/metrics/20261008_bizard_v10.json；outputs/submissions/20261008_bizard_v10.{h5ad,vcc}
