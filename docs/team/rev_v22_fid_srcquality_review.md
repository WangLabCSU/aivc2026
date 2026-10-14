# REV 裁定 — cb v2.2 fid 签名源质量加权 spec（四项全裁定 + 附加 C1–C4）

- 日期：2026-10-14；裁定人：REV；对象：`docs/team/cb_v22_fid_source_quality.md`
  （commit 85d19d5，cb-expert 撰写）；裁定送达 cb-expert 后入档。

## §7 四项逐条裁定

**① 认可**——A.1 加权公式链与 A.2 门控判据全部认可，三处预注册参数逐项裁定：
β=0.7（confv2 w 下限 0.7 同款软收缩，有入档先例）；τ=0.25（与 confv2/dirprior 显著门
一致，只沿用不新设）；LOPO ≥2pp 且双源（RPE1+HepG2）同时成立的 AND 门槛——比单源
判据更严，属只收紧方向，认可。A.1 关键不变量（权重作用于定显著集之前的 Δ，下游消费
Δ*）成立；与 iter014 四条闭环路线的四点区别（逐基因/数据驱动/前置于定显著/
source-LOPO 门控）成立，不构成重开已闭环路线。

**② 认可即收**——LOPO FAIL 则本线即收、记负结果入档、不追加第三轮。止损先例同
(b) 78a3f17；spec §5.2 失败路径条款（fid 维持 −0.018，与 scbasecount 同等待遇）一并锁死。

**③ 设计 B：REV 无异议，排期不裁**——B1（重叠 ≥50 靶/上下文才继续，不足如实上报即停）
与 B2（锚点仅作校准参考同表并排、不改冻结 λ 链）边界认可；affine 第 4 族参考点路径
（当前 n=3 refit 残差 ≤0.047）不触碰冻结项，认可。GPU 时段分配属排期权，归 team-lead
调度，REV 不代行；技术面无冲突点（(a)-v2 为 CPU n_jobs=40 训练，zeroshot 推理为 GPU）。

**④ 认可 cb 默认：adamson 不补建**——K562 系已占 3/7，补建推至 4/8 恰好加重本 spec
要修的 p53 轴机制（K562=TP53 缺陷，伪一致来源增量）；7 源对 LOPO（留出 2 源、5 源共识）
结构充足；9/30 方案名单与本地实况差异已由 §1 roster 澄清如实记录。若后续证据要求再另案送审。

## 附加 C 条款（随认可生效）

- **C1 权重冻结**：ρ 定义与 source_context_weights.parquet 在 LOPO 执行前冻结；不得在
  看到 LOPO 读数后回调 ρ/τ/β（防后验调参）。ρ 计算所用的模块谱快照列随权重表入档。
- **C2 池修正前置披露**：norman 重叠审计若 <10% 剔除，须在 LOPO 判定前入档说明
  （池定义修正非 FAIL）；不得据 LOPO 读数事后改池。
- **C3 接口落地回归**：gene_weights 键实现合入时，须重跑 no-key 5.27M 行逐位回归并入档
  （iter014 模板）；缺省逐位一致是带键路径启用前的硬前置。
- **C4 防回退对照**：若 LOPO PASS 且集成，bench 六分量全表重跑与 iter014 四条闭环负结果
  同表对照，随结果纪要入档（A.4 已承诺，升格为条款）；sign_fix 保持关闭、w 上限 1.0、
  reach/jac 反向守卫照 A.4 执行。

## 资产盘点抽查

通过：7 源签名 parquet 在档（另 tahoe_p12 可扩展）、confidence_v2.parquet 在档、
make_predictions.py 现有 sign_fix/dataset_weights 可选键（gene_weights 未存在，与
"新增"声明一致）、zeroshot 四上下文检查点在档、ntc_context_diagnostics.csv P53_axis 行
0.14/0.402/0.387/0.221/0.023/0.020 与 spec §0 逐字一致。

## 边界重申

本裁定不触发任何提交动作，λ 链/affine 系数/BizardNet 方向头不动，提交命名
bizard+版本号规则维持。GPU 时段分配权归 team-lead（PI）：(a)-v2 为 CPU n_jobs=40 训练，
zeroshot 推理为 GPU，无资源冲突。
