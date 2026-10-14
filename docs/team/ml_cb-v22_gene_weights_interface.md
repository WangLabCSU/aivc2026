# ml: cb v2.2 gene_weights 键合入 load_delta — REV C3 回归 PASS + 键路径单测 PASS

日期: 2026-10-14。执行方: ml-expert（team-lead 派单，spec docs/team/cb_v22_fid_source_quality.md §4）。

## 结论

1. **`gene_weights` 可选键已合入** `src/models/l1_signature/make_predictions.py` `load_delta`，
   键结构 `{file, sources, mode: "pre_significance"}`（spec 建议名，未改名）。
2. **REV C3 硬前置满足**: no-key 5,278,422 行逐位回归 PASS —
   `max|Δ|=0.0`、`n_bitwise_neq=0`、键集零差异（only0=only1=0）、无 NaN、
   report dict 逐键全等、快照 sha256 一致。键路径由此启用。
3. **键路径单测 7 组断言全过**（独立脚本，pytest 未安装）。

## 实现（make_predictions.py）

- 权重表 schema: parquet 列 `[source, gene, w]`；`sources` 可选过滤（缺省=表内全部源，
  未列源 w=1.0 中性）；`mode` 严格 `"pre_significance"`，否则 ValueError。
- 语义: 行级有效权重 = `dataset_weights` 源权重 × gene_weights (source→gene) 权重，
  走既有 Σw·Δ/Σw 加权均值路径（`if ds_w or gw_map` 触发）。
- w clip 到 **[0, 1]**（REV C4: 上限 1.0；负权=翻号属已闭环 sign_fix 路线，不启用）。
- 表外 (source, gene) → w=1.0 中性；**不作用于 generic 兜底行**（generic concat 在聚合后，无 `_w` 列）。
- report 仅在键激活时新增 `report["gene_weights"]` 块:
  `{file, mode, sources, table_rows, delta_rows_weighted, frac_neutral_1.0}`；
  no-key 路径 report 与基线逐键全等（回归验证）。

## 证据（outputs/gene_weights_regression/, git add -f）

| 项 | 值 |
|---|---|
| 基准 config | configs/l1_signature/dualsrc-deep-lam65.yaml（gwps+hct116_all, 无任何可选键） |
| 快照行数 | 5,278,422 |
| 快照 sha256 | 35c849049394af43...（snapshot_sha256.txt 全文） |
| compare 结果 | compare_result.json: `pass=true` 全字段（max_abs_delta 0.0 / n_bitwise_neq 0 / report_identical true / snapshot_sha256_match true） |
| 运行日志 | baseline_run.log / compare_run2.log / compare_run3.log（最终版代码复跑） |
| 回归脚本 | scripts/reg_gene_weights_nokey.py（baseline/compare 两相, exit 1 on fail） |

注: compare_run3.log 为**最终版代码**（含 report 块统计完善后）的复跑，PASS=True；
本次最终版改动仅触及 `gw_map` 非空才执行的统计分支与 report 键值，no-key 行为逐位不变（回归已证）。

## 单测（scripts/test_gene_weights_keypath.py，7 组全过）

1. no-key 基线等权均值（与旧聚合一致, report 无 gene_weights 键）
2. Σw·Δ/Σw 加权数学 + w clip（1.7→1.0, −0.2→0.0）+ 表外中性 + report 块
3. sources 子集（未列源 w=1.0）
4. dataset_weights × gene_weights 行级乘法组合
5. mode 守卫（非 pre_significance → ValueError）
6. 绝对路径解析
7. w=0 剔除源语义

合成签名 + monkeypatch ROOT，不触碰真实 data/。

## 边界确认

- 冻结未动: BizardNet 方向头、λ 链（λ47=bizard-v0.2）、affine 系数、gate_report。
- sign_fix 保持关闭；w ≤ 1.0；reach/jac 反向守卫 reserved（REV C4）。
- 真实权重表（cb source_context_weights.parquet, §A.1 ρ/Spearman 产出）落地后直接
  以 `{file, mode: "pre_significance"}` 接入，接口无需再改。
- consensus_delta L1.5 集成等 cb LOPO PASS + REV C1–C4 复审，本提交不含。
- 无提交操作（submission ops 归 team-lead）。
