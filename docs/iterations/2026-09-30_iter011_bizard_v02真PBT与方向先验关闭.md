# iter011（2026-09-30 夜间）: bizard-v0.2 真 PBT 首战超亲鸟；方向先验子集选择关闭

## 任务① bizard-v0.2（变异算子 + 真 PBT）

- 变异三维落地: pred_topk{off,150,500} / aux{0,.15,.3,.45} / 关 p53+IFN 模块列。
- **实测判负**: pred-topk 强烈有害且剂量依赖（150→Q −0.21, 500→−0.09, reach 崩）, 且 6 鸟全池
  soft-rank 聚合被稀疏零分位污染（Q 0.2031 vs 稠密 0.4110）; aux∈[.15,.45] 与模块开关中性。
- **真 PBT 有效**: 头鸟 b3_r → 末位 b4_s1/b6_g 继承+高斯扰动重训, 两只 gen2 鸟 R_Jurkat
  0.3904/0.3902 **均超亲鸟 0.3880**（rank_raw +0.3113/+0.3107 vs +0.3084）。
- 去毒种群聚合（b1/b3/b5+2×gen2, tag bz02_agg_v02dense）Q **0.4085** ≈ v0.1 0.4110（噪声内）。
- 结论: 发 2 维持 bizard_v0.1 提交物; v0.3 变异轴换特征级源 dropout/label-clip。

## 任务② fid 方向先验子集选择（dualsrc-deep λ3.5, Q_HepG2, 基线重验 0.3183≈ledger 0.3182）

- v1 全三分类（同向留/反向剔/单源显著 λ 减半）: **0.2527**（fid −0.101, reach −0.085, pds −0.035）。
- v2 仅反向剔除: **0.2961**（fid −0.024, reach −0.042, pds +0.001）——归因: 剔除本身即失分。
- **gate 未过, 路线关闭**（第三条: top-K 保强去弱 / 幅度缩水 / 方向先验）。机制统一:
  剔除条目 = 收缩可达 DE 集 → reach/fid-yield 塌。fid 的活杠杆只剩 λ 与官方斜率校准。

## 明日两发推荐

1. **发 1: dualsrc-deep λ4.5**（vcc 已 prep 全绿; 本地 Q 0.3306 vs λ3.5 0.3182, fid +0.041/
   pds +0.029/reach +0.014/jac +0.010, 仅 nmae −0.020; 官方预期 +0.05~+0.08, 兼 fid(λ) 斜率检验）。
2. **发 2: bizard_v0.1**（vcc 已 validate; fusion 家族 ensemble 独立性探针, Q 0.4110）。

产物: docs/bizard/v02_results.md ｜ outputs/metrics/20261003_bizard_v02_pbt.json ｜
data/interim/fid_dirprior/{dirprior_results,stats_*}.json ｜ bench tags bz02_*, fid_dp_*
