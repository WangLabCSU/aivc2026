# 2026-10-11 iter015：ensemble 官方首测 +0.0800、三源维度关闭、BizardNet v1.1 context-dropout

## 一、官方提交（UTC 10-11 唯一成功发）

**L1.9 ensemble 0.5\*heur+0.5\*fusion λ6.5**（entry `vxpevKCTFdjIs9Smp0ul`）：

| 指标 | 值 | vs 纯启发式 λ6.5 (+0.0886, Rank 633) |
|------|-----|------|
| overall | **+0.0800** | −0.0086 |
| rank | 765 | −132 |
| pds | 0.4839 | −0.018 |
| fid | −0.0530 | −0.019 |
| reach | 0.1112 | −0.016 |
| nmae | −0.0607 | +0.005 |
| jac | −0.0013 | −0.004 |

**判读**：
1. 官方 λ 移位律再验证——ensemble 本地峰 λ4.5（0.4073），按"官方峰≈本地峰+2"定位 λ6.5，+0.0800 落在 affine 预测带 (+0.05~+0.14) 内偏低端。
2. **关键反例：λ6.5 处 ensemble 本地 +0.0295（0.3181 vs 0.2886）但官方 −0.0086**——bench 的 blend 增益不迁移，与 fusion 家族同构（bench 对"学习式成分参与的组合"系统性乐观）。bench→official offset 排序：heur-deep −0.20 < ens −0.24 < fusion −0.28 < bizard-swarm −0.43。
3. 六指标全面小于纯启发式——**fusion 半程在官方一致拖累**；α 扫描的官方最优大概率 α<0.5，但该方向期望收益 ≤ +0.009，不值得再花额度。
4. **当前官方最优仍是 dualsrc-deep λ6.5 +0.0886（Rank 633）**。

## 二、三源 Δ̂ ensemble 扫描（heur × fusion × CCRN-loco × Bzn-skip，sd 匹配）

预检相关：heur-fusion r=0.616（组合增益来源）；ccrn-fusion r=0.888（CCRN 是 fusion 之子）；
bzn 幅度 0.008（其余 0.03，sd 匹配后可比）。9 点网格 @λ4.5（scripts/ens3_scan.py）：

| 权重 (h,f,c,b) | Q overall | | 权重 | Q overall |
|---|---|---|---|---|
| **2-way 0.5/0.5（基线）** | **0.4073** | | 0.45,0.45,0,0.10 (b1) | 0.4009 |
| 0.35,0.45,0.20,0 (s2) | 0.4018 | | 0.45,0.45,0.10,0 (c1) | 0.4006 |
| 0.40,0.40,0.10,0.10 (m1) | 0.4011 | | 0.40,0.40,0.20,0 (c2) | 0.3991 |
| 0.35,0.35,0.20,0.10 (m2) | 0.3987 | | 0.40,0.40,0,0.20 (b2) | 0.3972 |
| 0.45,0.35,0.20,0 (s1) | 0.3952 | | 0.30,0.30,0.40,0 (c3) | 0.3926 |

**全部 9 点低于 2-way 基线**。第三源维度关闭：CCRN/Bzn 的增量信息在 heur×fusion 凸包内已被覆盖，
且 bench 乐观偏差进一步放大其虚增。产物：scripts/ens3_scan.py、
data/interim/l15_fusion/v35/ensemble_scan/{delta_ens3_*,meta_ens3_*}、
outputs/metrics/local_bench/ens3_*/、anchor_normalized_ens3.json。

## 三、工程事件：UTC 10-11 额度异常

ensemble 出分后 chain 立即提交 bizard v10 → "allowance already used today (resets 00:00 UTC)"，
30 分钟后重试仍拒绝，但 `whoami` 显示 ready to submit。当日仅成功 1 发（ensemble）。
可能原因：队伍内他人占用、或出分后额度状态同步延迟。**处置**：bizard v10（λ39 moonshot）+
CCRN v0.1（λ3.5 清洁家族读数）顺延至 UTC 10-12 窗口（20:00 EDT）。两 vcc 均已 staged。

## 四、BizardNet v1.1：context-dropout（iter013 方向 a 落地）

- 补丁：`src/models/bizard/bizardnet.py` forward 训练时以 p 概率将整条上下文指纹替换为
  可学 null 向量（`train.context_dropout`，默认 0 → 与 v1.0 逐位一致）；`BIZNET_CFG` env 选 config。
- 动机：iter013 实证链——transformer 分量在指纹未见过的 hepg2 上是噪声（r=0.05），
  skip 头（r=0.345）才是迁移载体；context-dropout 阻断指纹记忆、强迫跨上下文路由。
- v1.1 = v1.0 超参不变 + context_dropout 0.2；workdir bizardnet11（train 数据 symlink 复用）。
- 判据：LOCO 臂 bench skip 载体 λ 扫描（12/16/18/20）超 v1.0（0.3407/0.3521/0.3427/0.3341）→ 替换明日 slot1；否则明日 slot1 仍 v10。
- 结果：（训练中，12ep ≈18min；后续回填）

## 五、明日（UTC 10-12）双发预案

- slot1：BizardNet（v1.1 若 bench 胜出，否则 v1.0 λ39）——bizard 家族 official 判定发。
- slot2：CCRN v0.1 λ3.5（`20261008_bizard_ccrn_v01.vcc` 已 staged）——清洁家族读数，
  决定 BizardNet v0.2 是否继续投入。
- 备胎：λ7.5-deep（fid 零穿越探针，官方期望 +0.088~0.089 边际；构建 30min 随时可发）。
- 不发：三源 ensemble（本地已负）、α 重扫（期望 ≤+0.009）。
