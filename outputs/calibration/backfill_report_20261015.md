# backfill_report_20261015 — 官方点回填与 affine refit（bizard family）

任务: comp-bio E1 建议 — 将每次官方提交制度化为 local→official affine 标定点。
仪器: `scripts/calib_append_official.py`（log FINAL_STATUS → OFFICIAL_POINTS 幂等追加 → refit）。
约束遵守: 未改任何生产提交代码路径（src/、configs/ 零触碰，git diff 可验）；sign_fix、λ、gate_report、PBT 奖励函数未动；未 git commit；未读 .h5ad/.parquet 大文件。

## 1. 追加点（3 条，OFFICIAL_POINTS 8 → 11）

| λ | model | entry_id | 日志 | 提交时间 (UTC) | Rank | Overall |
|---|---|---|---|---|---|---|
| 47 | bizard-v0.2 | BWy91Q4UXdusW6QcxjCU | logs/submit_lam47_1014.log | 2026-10-14T12:00:27Z | 908 | +0.045655 |
| 37 | bizard-v0.3 | SgXJeIQwv7taWP4B0GND | logs/submit_lam37_1015.log | 2026-10-15T03:52:47Z | 886 | +0.050214 |
| 35 | bizard-v0.4 | 58kzxoM9gGrF1QOdXS5k | logs/submit_lam35_1015.log | 2026-10-15T04:22:34Z | 887 | +0.049122 |

编辑备份: /tmp/official_alignment.py.bak.20261015_{014210,014238,014241}。每次追加后 compile 校验通过，fit_calib() 落盘 docs/research/official_alignment_v1.json（bizard family n=5/metric）。幂等性已验证（重跑打印"已在表内——幂等跳过"）。

## 2. per-metric 系数漂移（n=3 → n=5，official = a·local + b）

| metric | n=3 a | n=3 b | n=5 a (终) | n=5 b (终) | Δa | Δb |
|---|---|---|---|---|---|---|
| pds | +0.266014 | +0.135084 | +0.166822 | +0.187817 | −0.099 | +0.053 |
| nmae | +0.025748 | −0.012953 | +0.030254 | +0.017117 | +0.005 | +0.030 |
| fid | −0.723819 | +0.593986 | −0.977299 | +0.803362 | −0.253 | +0.209 |
| reach | +0.067196 | +0.045369 | +0.037691 | +0.054796 | −0.030 | +0.009 |
| jac | −0.854958 | +0.082409 | −1.001645 | +0.096785 | −0.147 | +0.014 |

（λ47 追加时 n=3→3 系数几乎不动：该点 iter021 已在 json 以近同值入库，Δa 量级 1e-5，仅 jac Δa +0.0124。上表以 n=3 起点（λ39/λ55/λ47）到 n=5 终态汇总。）

## 3. leave-one-out 误差（bizard family，逐指标）

| metric | LOO before (n=3) mean/max | LOO after (n=5) mean/max |
|---|---|---|
| pds | 0.0838 / 0.2346 | 0.0339 / 0.1180 |
| nmae | 0.0894 / 0.1544 | 0.0514 / 0.0867 |
| fid | 0.0003 / 0.0006 | 0.0136 / 0.0283 |
| reach | 0.0196 / 0.0489 | 0.0163 / 0.0495 |
| jac | 0.0058 / 0.0091 | 0.0018 / 0.0032 |

pds/nmae 的 LOO 随点数增加显著改善；fid 的 LOO 从近零略升（n=3 时三点几乎共线，0.0003 是过拟合假象，n=5 的 0.0136 更真实）。

**overall 级标定**（local avg_score → official Overall，5 发布点，含 λ39 机器快照 +0.0507、λ55 台账 +0.0330）:

```
official = 0.018477*local + 0.046835    n=5   R²=0.8244
LOO: mean|err| 0.0049, max|err| 0.0108 (n=5)
逐点残差: λ39 −0.0027, λ55 −0.0033, λ47 +0.0040, λ37 +0.0021, λ35 −0.0000
```

## 4. 对 local=0.3455（明日 v0.5 候选本地分）的官方期望预测

- **overall 级拟合: +0.0532**（推荐读数；LOO mean 0.0049 / max 0.0108 支撑）
- 敏感性: 若 λ39 按文档值 +0.0886 入拟合，则 official = 0.048089*local + 0.056180 → **+0.0728**。但该敏感性分支已被下述 §5 证伪，仅存档。
- per-metric 路线（5 指标预测取均值）系统性**高估** official Overall ~0.003–0.016：5 发布点回代残差全为负（λ39 −0.0156 / λ55 −0.0067 / λ47 −0.0031 / λ37 −0.0089 / λ35 −0.0114）。**提交前预测应以 overall 级标定为准**，per-metric 均值仅作区间上界参考。

## 5. 数据冲突标注（以日志/机器快照为准）

**λ39 官方 Overall 文档冲突**: iter021 台账与 rev_R1 引用 **+0.0886**，但该值实为同日对比对象——启发式家族冠军（heuristic λ6.5, Rank 633）的官方分（iter016 首次引作对比，iter020 台账 line 61 误植到 bizard 曲线）。bizard 家族 λ39 真实官方分为 **+0.0507**（Rank 884，entry 1I1bUHfcecnbACv4X0Bc，机器快照 docs/research/20261008_bizard_v10_online.json）。λ39 无 FINAL_STATUS 日志，机器快照为最强证据，故以 0.0507 入 overall 拟合。

**由此修正官方 λ 曲线**（bizard 家族，n=5）:
**λ39 +0.0507 > λ37 +0.0502 > λ35 +0.0491 > λ47 +0.0457 > λ55 +0.0330**

仍自 λ39 起单调递减（峰在 λ39 或以下），但 iter021"下探期望官方分 ≥ +0.09"的读数不成立——λ39/λ37/λ35 三点挤在 +0.049~+0.051，曲线在 λ39 以下近乎平坦（λ37、λ35 与 λ39 差 <0.0015，Rank 886/887 vs 884）。昨日 λ<39 双发的实测结果即本次回填的 λ37/λ35 两点：**下探收益已枯竭**，若继续二分 λ30/λ33 预期收益 ≤0.002，不值得再耗 slot。对 local=0.3455（v0.5 候选，高于全部已发布点的 local），overall 拟合外推 +0.0532 是当前最乐观的诚实估计；预测区间建议按 LOO max 0.0108 放宽至 +0.042 ~ +0.053。

## 6. 生产链触碰裁定

**分析侧仪器，允许写表。** scripts/official_alignment.py 与 docs/research/official_alignment_v1.json 在 src/ 下零引用；生产提交链的 affine 是另一冻结系统（configs/bizard/v02.yaml affine_a=2.0958 / affine_b=0.5259，经 src/models/bizard/swarm.py PBT fitness 使用）——两者零交集，本次 refit 未改任何生产参数。任务条件停止规则不触发。

## 7. 仪器附注

- `scripts/calib_append_official.py` 已修复 loo_errors 的指标元组硬编码缺陷（现按 rows 中实际出现的 metric 迭代），overall 级 LOO 行可正常打印。
- mse 永不入拟合：官方 mse 恒 0，本地 anchor 因 `_norm` 后缀永不解析出该指标。
- local tag 命名遗留: bzbiz10skip_lam16 实为 λ39 的 artifact（历史命名），报告与脚本注释均已标注。
