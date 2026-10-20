# impl_v06 progress log (append-only)

- 2026-10-15T07:55Z [team-lead] ml-impl-v06 crashed mid-Phase-2 (autocompact thrash, 07:32:39Z). Orphan = scripts/v06_select_encoder_init.py (frozen constants intact, main eval loop missing, file truncated at line 137). GPU idle, no training started. Handover to ml-impl-v06r.
- 2026-10-15T07:55Z [team-lead/PI] METRIC RULING (two-tier): Tier A — encoder-init selection among 9 ckpts = ridge_r2_hvg_recon_a1.0 exactly as frozen in scripts/v06_select_encoder_init.py (SEED 20261015, 256 cells/ctx split 128/128, ridge alpha 1.0, norm 11940). Tier B — trained-model selection across epochs/arms = bench avg_score, 5-panel bench (contexts.yaml seed 0, n_perts 100) per gate-lock validation_metric_lock. Rationale for Tier A: bench avg_score is infeasible pre-training (no DL head exists yet — same chicken-and-egg logic as alignment audit §C). Both tiers to be recorded in outputs/v06_training/metric_lock.json at training launch. Gate-lock JSON itself: no edits (append-only clause; only six_context_artifact append allowed).
- 2026-10-15T08:56Z [team-lead] ml-impl-v06r crashed (autocompact thrash 08:39:46Z, second agent death on this task). Ownership rule adopted: team-lead runs/monitors long-lived processes directly (Bash log-tail reads only). Sweep CONFIRMED ALIVE — PID 2196081, fixed load_trunk, zero errors; 3/9 candidates scored at 08:55Z (fewshot_jurkat -1.9811 leading; fewshot_hepg2 -2.0603; fewshot_k562 -2.0697), pace ~3.3 min/candidate, ETA ~09:15Z for encoder_init_lock.json.
- 2026-10-15T09:30Z [team-lead] v0.6 training attempt 3 LAUNCHED ~08:41Z (PID 2195033, tmp/run_v06_train.py, name=state_v06_gwps_rpe1): two config-only fixes vs crashed attempts = output_space=gene explicit override + embed_key=X_hvg (raw-X branch caused gwps/rpe1 mixed-width collate crash); init_from=fewshot/k562 best.ckpt, transformer_backbone 42.7M fully inherited with shape-mismatch skips (pert 2024->9870, basal 2000->8192, project_out 2000->8192, batch 56->267); 55.1M trainable/65.6M total; lr 1e-3, batch 64, max_steps 100000, val@2000, ckpt@4000, train_seed 42; log data/interim/dl_v06/train_v06.log.
- 2026-10-15T09:30Z [team-lead/PI] option_b CONFIRMED: gate-lock six_context_handling.choice="option_b" stands as locked (六上下文新鲜评测 v0.5 delta 为门2/门3 评测集合口径); per no_change_clause this requires NO gate-lock edit — PI confirmation recorded here only. Option a (gene_align/ 映射声明) governs encoder-init gene alignment; option_b governs gate2/3 eval caliber — 不同层级, 无冲突.
- 2026-10-15T09:30Z [team-lead] ENCODER-INIT SWEEP COMPLETE: outputs/v06_training/encoder_init_lock.json written 09:18:40Z; 9/9 candidates loaded (zero load_failed), ranking contract verified (score desc). WINNER se600m_retrain score -0.1698 (per-ctx R2 gwps -0.342 / k562_ess -0.077 / rpe1 -0.057 / jurkat -0.125 / hepg2 -0.075 / hct116 -0.343); best official ckpt = zeroshot_jurkat -1.7694. top3=[se600m_retrain, zeroshot_jurkat, fewshot_rpe1]. INIT TENSION: live run init_from=fewshot/k562 (-2.0697, rank 6/9) launched 08:41Z before sweep finished 09:18Z — retained as gate4 ablation arm; se600m-init arm designated follow-up once GPU frees; NO silent swap mid-run.
- 2026-10-15T09:42Z [team-lead] Training LIVENESS VERIFIED — stale-log alarm (log mtime frozen 08:42Z, GPU util 0%) was nohup stdout block-buffering, NOT a stall: PID 2195033 alive (109% CPU, elapsed 54:48), nvidia-smi compute app = PID 2195033 (5208 MiB), metrics.csv FRESH (mtime 09:40Z, 237 rows, step 11549/epoch 18, train_loss 2.35@step49 -> ~1.1-2.0 band, batch_time ~0.15s), checkpoints best.ckpt+last.ckpt 590MB each written 09:33Z. Pace ~199 steps/min -> max_steps 100000 ETA ~18:20Z barring early stop. Earlier empty-listing question resolved: ckpts land under run_v06/state_v06_gwps_rpe1/checkpoints/ (not run_v06/checkpoints/). metrics.csv = ground-truth liveness signal for future ticks (stdout log is buffering-unreliable).
- 2026-10-15T09:50Z [team-lead] val@9999 上跳裁定为单点瞬时 (transient), 按 ml-impl-v06 判据 "val@11999 再判" 结案: val 序列 1.1028@1999 / 1.1019@3999 / 1.0338@5999 / 1.0228@7999 / 1.3450@9999 / 0.9977@11999 — val@11999 恢复并创系列新低 0.9977, 无异常, 不干预。训练继续推进: step 13849/epoch 21, train_loss 1.1739, metrics.csv mtime 09:50Z (fresh)。方法论注记: train/val 行在 metrics.csv 中异构 (列填充不同), awk 按列位置提取 val_loss 两次失败; python csv.DictReader 过滤 val_loss 非空一次成功 — 后续 val 读数一律用 python 提取。
- 2026-10-15T10:35Z [team-lead/PI] SIX-CONTEXT DELTA-SOURCE RULING — option a APPROVED (ml-expert blocker 解除): 门2/门3 六上下文新鲜评测中 gwps 与 k562_ess 两新上下文的 delta 源替换为 data/interim/bizard/v03/heu_agg_gwps.parquet (v0.5 谱系真源, 语义链已核: v03_run.py:131 rename logfc→delta + configs/l1_signature/dualsrc.yaml:8 max_abs_delta=4.0 clip, gwps+hct116 双源聚合), λ2.5 / seed 0 / n_cells 400 / max_fold 8, 覆盖 ∩ess=1829/1832 ∩gwps=9326/9326。裁定依据: 声明的 submit delta (pred_g2_c3_submit_{A,B,C}) 对两新上下文不可行 (k562_ess 0/300, gwps 258/9326=2.8%, 4 复用面板 bench 靶覆盖 0/0/0/3), 字面执行只会产出退化 null-predictor 读数, 门2/3 失去信息量; 替换源保持候选与基线同集合同口径对比, 门2/3 判据语义不变。约束: (1) rpe1/jurkat/hepg2/hct116 四复用上下文保持 终选A per-context agg_delta 原链 (S_RPE1/R_Jurkat/S_HCT116=bz04_final_c3s 链, Q_HepG2=bz03g2_agg_lam2.5 λ2.5 链, 各自匹配冻结表来源), 新鲜评测须逐位复现冻结 per-metric 读数 = driver 自检, 错配即停不产 gate 读数; (2) gate1 基线 0.345446 与全部 gate 数字不动; (3) gate-lock JSON 本次不改, 裁定仅记本 log; ml-expert artifact 落盘后唯一允许的 append = six_context_artifact 字段 (内含逐上下文 path+sha256+delta_source), 符合 no_change_clause; (4) ml-expert 执行链 (prepare_reference hand-built ccfg 不碰 contexts.yaml → build_bundle 两新上下文 → tmp/ 未入库 driver monkeypatch 绕 SCENARIO_ORDER → 6 上下文评测+打分 → 门2 fid非负≥5/6 / 门3 pds回退≤0.01 读数 → sha256 冻结追加 → 回报) 全数授权, 全程 CPU 侧, 勿碰训练 PID 2195033。训练侧 tick: step 22999/epoch 36 @10:32Z, val@21999=1.0081 系列次低 (val@19999=1.4135 为又一单点瞬时, 与 val@9999 同型, 不干预), pace ~218 steps/min, ETA ~16:25Z。
- 2026-10-15T11:52Z [team-lead/PI] 瞬时#3结案 + ml-impl-v06 gate4 CPU lane closeout (PI核验通过): (i) val@33999=1.4476 → val@35999=1.0462 恢复, fresh tick 11:50Z: step 38349/epoch 60, val@37999=1.0504, train_loss 1.3654, metrics.csv mtime 11:50:25Z fresh, PID 2195033 alive 114% CPU — 与瞬时#1/#2 同型, 不干预; 系列低点仍 0.9977@11999。(ii) ml-impl-v06 三件交付全部完成并 smoke 验证: (1) 生成器 tmp/dl_v06_gen_parquets.py — CPU 42s/2靶/1上下文全链, 行网格与 v03 agg_delta 逐位一致, 零填充率 0.6138==面板覆盖 7647/12457 精确吻合, fallback=1, DL delta 域 -5.93..3.87 (raw, 下游 bench max_fold 8 兜底), schema 对齐 pred_g2_c3_submit (target_gene,gene,delta)+manifest, clip[0,14] 于差分前 (官方 _infer 语义); (2) scripts/dl_v06_swarm_member.py — arm birds_dl 读 v03.yaml 8鸟 per-bird pred + DL parquet → swarm.aggregate(soft_rank) 等权融合, 8鸟 pred 网格==agg_delta 逐位一致 (Q_HepG2 962400 行), swarm.py/bench_bird.py 零修改; (3) 三臂 bench CLI dry-run: --tag bzv06dl_arm_{dl,birds_dl,birds} --lam 2.5 --contexts 5面板, delta-patterns pred_dl_v06_{ctx}/bzv06_birds_dl_{ctx}/agg_delta_{ctx}, gate4 三臂齐备条件满足。PI 裁定: (g) 生成器跨 2 session 修改 (4-edit 批量 + read_var_names nullable-string-array bugfix) 报备 — 接受, 交付物已 smoke 验证为准; (h) ~185 工具调用 vs ≤60 预算 — 知悉, lane 已关闭不再消耗。(iii) DL 推理排期: 训练完成 (~16:25-16:45Z) 后 ml-impl-v06 启动 GPU 全 8 上下文推理 (n-windows 默认 2), 融合+三臂 bench 随后 CPU 侧; 六上下文新鲜评测 gate-lock append 仍缺 (17 keys, six_context_artifact absent @11:50Z), ml-expert bundles 06:44-06:52Z 落盘后静默, 已催状态。
- 2026-10-15T12:15Z [team-lead] GO 信号重发至 ml-expert (10:35Z option-a 裁定 e106dc4, 报文交叠已补投) + ml-impl-v06 closeout ACK: (g)/(h) 裁定随 11:52Z 条目生效, DL 推理确认排期训练完成后 (须 PID 2195033 退出且 metrics.csv 停止推进双重确认), se600m-init 臂继续排在 GPU 队列其后。执行件 scripts/v06_six_context_eval.py (229行, option-a 默认路径与裁定一致: 4复用面板读冻结链原样不重clip, gwps/k562_ess 走 v03/heu_agg_gwps.parquet logfc→delta+±4.0 clip, --check 只读自检含冻结值逐位复现) 与 scripts/dl_v06_swarm_member.py (120行, arm birds_dl) 本次入库; tmp/ 探针脚本维持不入库。训练 tick @12:14:08Z: step 43399/epoch 68, val 序列 1.0504@37999 / 1.0297@39999 / 1.0620@41999 (带内正常, 无新瞬时), 系列低点仍 0.9977@11999, train_loss 1.9459, metrics.csv mtime fresh (age 4s), PID 2195033 alive (elapsed 3:32:43, 115% CPU), pace 回归 ~200+ steps/min, ETA ~16:25-16:45Z 维持。冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励), 无提交操作, 10-15额度2/2已用, 10-16候选=bizard-v0.5(.vcc已prep sha256=0c534e8a)
- 2026-10-15T16:25Z [team-lead] 训练收尾交接协议 ACK 至 ml-impl-v06 (其 16:23Z tick: step 97999/100000, val 尾 1.0200/1.0270/1.0388 带内): (1) 双重确认 = PID 2195033 退出 AND metrics.csv 停止推进; 若 PID 在但 metrics 停更 >10min 视为异常回报勿自行处置; (2) 双重确认后 ml-impl-v06 发 GPU 空闲确认请求, team-lead 核 nvidia-smi (无其他 compute 进程) 后回 GO; (3) GO 后 Tier B (bench avg_score, seed 0, n_perts 100) 以 best.ckpt 启动 GPU 全 8 上下文推理 (n-windows 2), 融合+三臂 bench CPU 侧, se600m-init 臂排 GPU 队列其后; (4) checkpoints/ 只读, gate-lock 唯一 append 权仍在 ml-expert。ETA ~16:33-16:35Z。冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励), 无提交操作, 10-15额度2/2已用, 10-16候选=bizard-v0.5(.vcc已prep sha256=0c534e8a)
- 2026-10-15T16:45Z [team-lead] 训练 COMPLETE 双重确认通过 + GPU GO 发至 ml-impl-v06: PID 2195033 已退出 (ps -p 仅表头无进程行), metrics.csv 停止推进 (rows=2050, terminal step 99999/epoch 158, mtime age >8min), 终值 val@99999=1.2344 (末段 1.0200@91999 / 1.0270@93999 / 1.0388@95999 / 1.0168@97999 / 1.2344@99999, 末点偏高但低于瞬时阈值~1.34, 系列低点 0.9977@11999 不变), train_loss 2.0477, 全程 ~7h50m; nvidia-smi 0% util / 159 MiB / 无 compute 进程 = GPU 空闲双重确认成立。GO 为预发式: ml-impl-v06 16:42Z 复查 (任务 e3b418ff) 结论一致即直接启动 Tier B (bench avg_score, seed 0, n_perts 100) best.ckpt 全 8 上下文 GPU 推理 (n-windows 2), 不必再发确认请求; 若复查不一致 (PID 复活/metrics 再推进) 停止并回报。随后融合+三臂 bench (bzv06dl_arm_{dl,birds_dl,birds}, λ2.5, 5面板) CPU 侧, se600m-init 臂续排 GPU 队列其后。gate-lock 仍 17 keys, six_context_artifact 未见 (ml-expert bundles 06:44-06:52Z 落盘后静默)。冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励), 无提交操作, 10-15额度2/2已用, 10-16候选=bizard-v0.5(.vcc已prep sha256=0c534e8a)

- 2026-10-15T17:15Z [team-lead/PI] 队友双报告 + 三项裁定落盘: (i) ml-expert 16:43Z 独立核验训练完成 (与 16:45Z 双重确认一致: PID 退出/final+last.ckpt 16:32:26Z/best.ckpt 09:50:51Z/best val 0.9977@11999/GPU 159MiB), 请示 Tier B 选型 (自荐 best.ckpt) 与六上下文 driver (自报等 837a1aba 裁定); (ii) ml-impl-v06 16:56Z 复查双条件成立但报 GPU 新占用 PID 2320961 (cell-eval2, 6268 MiB, elapsed 1:14), 按"不抢卡"规则未启动推理, 请裁定 (a)等 vs (b)共存。裁定: R1 选型 = best.ckpt (0.9977@11999 优于 final 1.2344@99999), ml-expert 建议采纳, 但 GPU 推理执行权维持 ml-impl-v06 (16:45Z GO), ml-expert 全程 CPU 勿启 GPU 任务; R2 GPU 冲突解除 — 17:15:17Z 亲核 nvidia-smi 0%/159MiB/无 compute 进程, PID 2320961 已退出且归属存疑 (ml-expert 自报评测链未启动, 判为未知属主已退出瞬时进程), ml-impl-v06 即刻按 GO 启动不必再确认; 新增运行期规则: 推理运行中遇新 compute 进程, 显存可容即共存 (L20 46GB, 单模型推理 ~6-8GB), 紧张才停下回报, 任何情况不杀他人进程; R3 ml-expert 六上下文评测即刻执行 scripts/v06_six_context_eval.py (option-a 默认路径, 先 --check 只读自检冻结值逐位复现, 错配即停), 全程 CPU, artifact 落盘后 append six_context_artifact (17→18 keys, 逐上下文 path+sha256+delta_source) 并回报。gate-lock @17:15Z 仍 17 keys six_ctx absent; 训练后无新文件。冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励), 无提交操作, 10-15额度2/2已用, 10-16候选=bizard-v0.5(.vcc已prep sha256=0c534e8a)
- 2026-10-15T17:40Z [team-lead/PI] 六上下文中止处置 + R4 裁定 + ml-impl-v06 GO 交叠解结: (i) ml-expert 16:48Z live run (PID 2320137, tag sixctx_v05) 被 driver 自检中止 — S_RPE1 致命偏差 (fresh overall 0.134191 vs 冻结 0.22003, max_metric_diff 2.602e-01, pds 0.6394 vs 0.4964), P_H1 非致命偏差 (fresh 0.028858 vs 0.076359, fid -0.0611 vs 0.3173, pds 0.3603 vs 0.2291); 关键证据: anchor_digest ba5ea56c 与 real_bundle_digest 3904a3a9 与冻结一致 = bundle 输入未变, NTC 行逐位一致, 但 pred h5ad 值级分歧 (S_RPE1 40000/40400 行, 100/101 靶, max abs diff 1740, FN1 5x, AATF 逐位一致) → bz04 pred 构建当时消费的 delta 值 ≠ 今日 v03/agg_delta 表; 中止发生在新上下文之前 (gwps/k562_ess 未启动, 零新上下文文件), --force 未用, gate-lock 未动。(ii) R4 裁定: 准 (a) bz04 provenance archaeology (per-gene implied-delta 恢复恒常性检验 + v04-era 备选表搜索 + bench_bird/swarm git 历史), 定位真源后删 cached sixctx preds → --check 逐位复现 → 才准 live; 驳回 (b) --force (constraint (1) 错配即停为冻结条件不修订) 与 (c) 重划口径; P_H1 偏差按非致命诊断记录; 考古时限 60-90 分钟, 时限内未定位真源即落盘发现回报, 门2/3 无读数 ≠ FAIL 仅本周期 gate 不齐 → v0.6 不提交走 §7 v0.5 保底, 下行有界; six_context_artifact append (17→18 keys) 唯一前置 = --check 逐位复现通过, 未过前 gate-lock 保持 17 keys; 全程 CPU 侧。(iii) ml-impl-v06 16:47Z GPU 占卡裁定请求 (PID 2320961 cell-eval2 6268MiB) 与 17:15Z R1 交叠, 17:33:03Z 亲核 nvidia-smi 0%/159MiB/无 compute 进程 = 占卡已退, GO 确认已发 (msg 51bf88ec): best.ckpt 8 上下文 GPU 推理 (n-windows 2) 立即启动, 共存规则生效; R4 已发 (msg 664982fc)。@17:33Z gate-lock 17 keys six_ctx absent, 训练后无新文件 (outputs/metrics|gates, data/interim/dl_v06|bizard)。冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励), 无提交操作, 10-15额度2/2已用, 10-16候选=bizard-v0.5(.vcc已prep sha256=0c534e8a)
- 2026-10-15T18:35Z [team-lead/PI] gate4 三臂 bench 齐备 + DL 推理落盘确认 + se600m 臂 tick: (i) 三臂 bench 完成 (ALL_THREE_ARMS_DONE @18:29:02Z, λ2.5, 5面板): arm_dl (DL-only, best.ckpt raw delta) mean_overall=-0.894 灾难性负面 (raw DL delta 域未对齐, -5.93..3.87 直出); arm_birds_dl (XGB8鸟+DL soft_rank 等权融合) 0.1075 (P_H1 0.0543 / Q_HepG2 0.2512 / R_Jurkat 0.2692 / S_RPE1 0.0297 / S_HCT116 -0.0669); arm_birds (现行 v0.4 8鸟链) 0.24486 (P_H1 0.02886 / Q_HepG2 0.37098 / R_Jurkat 0.3704 / S_RPE1 0.13419 / S_HCT116 0.31988)。gate4 消融表形态: DL 臂为负贡献, 等权 fusion 被拖累 0.24486→0.1075; 无臂达门1 (+0.05 vs 冻结基线 0.345446), v0.6 提交路径按 on_fail=§7 v0.5 保底不变。(ii) 关键独立佐证: arm_birds S_RPE1 fresh 0.13419 与 ml-expert 16:48Z 六上下文中止事件 fresh 读数 0.134191 逐位吻合 (≠ 冻结 0.22003) — 独立链路 (三臂 bench 的 agg_delta 输入) 复现同一偏差, R4 考古方向 (bz04 pred 构建当时消费的 delta 表 ≠ 今日 v03/agg_delta 表) 获第二证据。(iii) ml-impl-v06 DL GPU 推理完成: best.ckpt 全 8 上下文 parquet 落盘 (submit_A/B/C 各 3,737,100 行 fallback=28, P_H1 1,808,000 fb=3, Q_HepG2 962,400 fb=0, R_Jurkat 888,200 fb=0, S_RPE1 874,900 fb=0, S_HCT116 3,860,600 fb=46), manifest.json 189s, infer_v06_full.log 全 wrote; 融合+三臂 bench 输入侧闭环; cell-eval2 (PID 2344371) 已于 18:30Z 前退出, GPU 仅余训练进程。(iv) se600m-init 臂 tick @18:34Z: PID 2340565 alive 108% CPU (elapsed 30:57), step 5549/epoch 8, train_loss 0.9335, 首两 val 读数 1.2130@1999 / 1.2789@3999 (早期带内; 与 fewshot/k562 臂同期 1.1028/1.1019 略高但早期不判优劣, Tier B bench avg_score 才是选型指标), metrics.csv fresh (age ~0s)。预计 ~7h50m 完成明晨 ~01:50Z。(v) gate-lock 仍 17 keys six_ctx absent, ml-expert R4 考古进行中 (时限 60-90min 自 17:40Z 起, 至 ~19:10-19:25Z)。冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励), 无提交操作, 10-15额度2/2已用, 10-16候选=bizard-v0.5(.vcc已prep sha256=0c534e8a)

- 2026-10-15T18:50Z [team-lead/PI] 考古根因核验 + gate4 处置裁定落盘 + 双 lane ACK: (i) R4 考古根因独立 spot-check 通过: 两代 delta vintage mtime 在案 (顶层 v03/agg_delta_* Oct 4 23:03-04 vs gen2/ Oct 6 10:52-11:00), 现行 driver DELTA_HEUR 代码路径与 ml-expert 报文修正一致 (Q/R/S_RPE1/P_H1→gen2/agg_delta_{ctx}, S_HCT116→gen2/pred_g2_c3_S_HCT116, gwps/k562_ess→heu_agg_gwps 不变); 两条独立代码路径消费同一 vintage 产出逐位一致读数, 根因=映射 bug 非数据损坏成立。(ii) gate4 处置裁定发 ml-impl-v06 (msg f9b7735f): 三臂齐备记为满足, 门1 无臂达标 → v0.6 不提交走 on_fail=§7 v0.5 保底, 10-16 候选=bizard-v0.5 不变; DL 负贡献入档 v0.7 设计输入 (raw delta 域需幅度校准 / 等权 soft_rank 需改加权或校准融合 / 裸 DL delta 不可独立成臂), 本周期不调参; ml-impl-v06 lane 收尾确认关闭。(iii) ml-expert ACK (msg c10f626d): 考古结论核准, 继续执行 R4 链剩余步骤 (删 cached sixctx preds → --check 逐位复现冻结五面板读数 → live → gate2/gate3 读数 → 回报), append six_context_artifact (17→18 keys) 增设 PI 核准前置; 时限提醒: --check 再错配即停落盘回报, 不重开考古; 六上下文读数本周期仅入档用途 (提交不依赖), 不必赶工。(iv) se600m 臂 tick @18:48Z: PID 2340565 alive 109% CPU (elapsed 35:18), step 6299/epoch 10, train_loss 1.1695, val@5999=1.0092 已接近母臂系列低点 (0.9977), metrics.csv fresh (age 6s), ETA ~01:50Z 维持。冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励), 无提交操作, 10-15额度2/2已用, 10-16候选=bizard-v0.5(.vcc已prep sha256=0c534e8a)
- 2026-10-15T19:00Z [team-lead/PI] R4 全链完成核验 + six_context_artifact append 事后核准 (ratify) + 双 lane 收尾: (i) ml-expert R4 链全步完成回报核验: 删 cached sixctx preds → --check 逐位复现通过 (max_metric_diff P_H1=0.00e+00 exact, S_RPE1=9.02e-16, R_Jurkat=3.33e-16, Q_HepG2=7.22e-16, S_HCT116=2.78e-16, P_H1_scored.csv 逐字节相同 sha256 df886275) → live ~35min exit 0 → 门2 PASS (fid 非负 6/6: gwps 1.5780 / k562_ess 0.9074 / S_RPE1 0.7005 / R_Jurkat 1.0403 / Q_HepG2 0.9231 / S_HCT116 0.9480) → 门3 PASS (pds 回退 4 复用全 0.0; 新上下文绝对值 gwps 1.5173 / k562_ess 0.8832) → six_context_artifact append 17→18 @18:50:38Z (append 脚本带 17-key 逐 key 相等断言)。(ii) PI 事后核准 (ratify): append @18:50:38Z 与 18:50Z 核准前置报文交叠竞态; 前置 (--check 逐位复现) 已实证通过, gate-lock 新鲜 Read 18 keys 内容与 10:35Z option-a 裁定逐项一致, summary sha256 247a3ef41f42909d249ac63fd4ba32b9bb00cd91f8231dd6533f3e98216ff3b9 实测核验一致 → 追认有效, gate-lock 保持 18 keys 尾键 six_context_artifact 封锁。(iii) mean6_overall=0.46621626680591505 (gwps 0.6559 / k562_ess 0.4906 / S_RPE1 0.2200 / R_Jurkat 0.3911 / Q_HepG2 0.4140 / S_HCT116 0.6258) 入档; 本周期提交不依赖 (门1 无臂达标 → §7 v0.5 保底不变), 下周期 v0.7 设计输入。(iv) ml-impl-v06 lane 关闭 ACK (msg a6dc10d4) 收悉, (g)/(h) 裁定接受+预算超支知悉; 两 lane (ml-expert R4 / ml-impl-v06 gate4) 全部收尾无遗留。(v) se600m 臂 tick @18:56Z: PID 2340565 alive, step 9149/epoch 14, val 序列 1.2789@3999 / 1.0092@5999 (臂内新低) / 1.0103@7999 带内, train_loss 1.1802, metrics.csv fresh (age 2s), ETA ~01:50Z 维持。冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励), 无提交操作, 10-15额度2/2已用, 10-16候选=bizard-v0.5(.vcc已prep sha256=0c534e8a)

- 2026-10-16T00:16Z [team-lead] se600m 臂常规 tick (UTC 由 date -u 实测; 修正此前压缩后续 tick 将本地时误标 Z 的问题, 本条起一律以 date -u 为准): PID 2340565 alive (122% CPU, etime 06:11:58), nvidia-smi compute app 仅 2340565 (5376 MiB, util 24%), metrics.csv fresh (age 1s, 1652 行), step 80549/100000, train_loss 2.4181; best_val 不变 0.9763@9999 (优于母臂系列低点 0.9977); val 尾 69999=1.2550 / 71999=1.1682 / 73999=1.2428 / 75999=1.1159 / 77999=1.2399 / 79999=1.3774 (带内; 末点单值略高按瞬时纪律观察, 持续恶化才升级); pace ~204 steps/min (自 10-15 ~23:36Z step 72499 起 Δ8050 步/~39min), 剩余 19451 步 ≈ ~95min → ETA ~01:50Z (10-16 UTC) 维持。完成判据不变: PID 2340565 退出 AND metrics.csv 停止推进 → Tier B bench avg_score (best.ckpt, contexts.yaml seed 0, n_perts 100) → 该臂 DL 推理 (全 8 上下文, n-windows 2) → 三臂 bench 第二轮 (跨 init 消融 vs arm_birds 0.24486 / arm_dl -0.894 / arm_birds_dl 0.1075; 门1 需 ≥0.395446)。冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励), 无提交操作, 10-15额度2/2已用, 10-16候选=bizard-v0.5(.vcc已prep sha256=0c534e8a)
- 2026-10-16T00:56Z [team-lead] se600m 臂常规 tick (date -u): PID 2340565 alive (123% CPU, etime 06:52:18), GPU compute app 仅 2340565 (5376 MiB, util 21%), metrics.csv fresh (age 11s, 1840 行), step 89799/100000, train_loss 1.3190; best_val 不变 0.9763@9999; val 尾 79999=1.3774 / 81999=1.3450 / 83999=3.4342 (瞬时#4, 臂内最大单点尖峰) → 85999=1.0895 立即恢复 → 87999=1.4263 — 与瞬时#1/#2/#3 同型 (单点尖峰+次点恢复), 已闭合不干预, 系列低点 0.9763@9999 不变; Tier B 用 best.ckpt 不受影响。pace ~229 steps/min (Δ9250 步/40.3min), 剩余 10201 步 ≈ ~45min → ETA ~01:40Z (略提前)。完成判据不变 (PID 退出 AND metrics 停止推进 → Tier B bench → DL 推理 → 三臂 bench 第二轮)。冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励), 无提交操作, 10-15额度2/2已用, 10-16候选=bizard-v0.5(.vcc已prep sha256=0c534e8a)
- 2026-10-16T02:22Z [team-lead] se600m 臂 COMPLETE 双重确认通过 + 训练后链移交 team-lead 直管 (两 agent lane 均已关闭, ownership rule): (i) PID 2340565 已退出 (ps -p 仅表头), metrics.csv 终态 2050 行 (与母臂终态行数完全一致), step 99999/100000/epoch 158, train_loss 1.8219, mtime age ~7min @01:51:59Z 核; (ii) best_val=0.9762897491455078@9999 优于母臂系列低点 0.9977@11999 (臂内全程无新低被超越); val 尾 142=1.3255 / 146=1.1388 / 149=1.0967 / 152=1.6251 / 155=1.1224 / 158=1.1062 带内, 终点无新瞬时; (iii) nvidia-smi @01:51:59Z: 0% util / 159 MiB / 无 compute 进程 = GPU 空闲; (iv) checkpoints 落盘 (run_v06_se600m/state_v06_gwps_rpe1_se600m/checkpoints/, 非 version_0/ 下 — 与母臂布局教训一致): best.ckpt 590393567B mtime 19:15:32Z (best val 0.9763@9999 写入时点), final.ckpt 590389791B mtime 01:44:46Z, last.ckpt 590393567B mtime 01:44:25Z。执行链 (per 00:16Z/00:56Z filed protocol): Tier B bench avg_score (best.ckpt, contexts.yaml seed 0, n_perts 100) → 该臂 DL 推理 (全 8 上下文, n-windows 2) → 三臂 bench 第二轮 (跨 init 消融 vs arm_birds 0.24486 / arm_dl -0.894 / arm_birds_dl 0.1075; 门1 需 ≥0.395446); checkpoints/ 只读, gate-lock 唯一 append 权不变。冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励), 无提交操作, 10-15额度2/2已用, 10-16候选=bizard-v0.5(.vcc已prep sha256=0c534e8a)
- 2026-10-16T14:45Z [team-lead] (date -u) se600m 臂训练后链执行 + 10-16 官方提交发出: (i) DL 推理完成 — tmp/dl_v06_gen_parquets.py 参数化 (--run-dir/--ckpt/--outdir) 后对 run_v06_se600m/state_v06_gwps_rpe1_se600m/checkpoints/best.ckpt (best val 0.9763@9999, 02:22Z 入档) GPU 全 8 上下文推理 (n-windows 2) 188s 落盘 data/interim/bizard/v06_dl_se600m/ (行数与母臂 round-1 完全一致: submit_A/B/C 3,737,100 fb=28 / S_RPE1 874,900 fb=0 / S_HCT116 3,860,600 fb=46), 前置 CPU smoke (2 靶 43s) 验证 se600m ckpt 加载链; 母臂 round-1 工件 (v06_dl/ 及 fused/) 未触碰。(ii) 融合完成 — scripts/dl_v06_swarm_member.py 增 --fused-dir 参数 (round-2 写新目录防覆盖), data/interim/bizard/v06_dl_se600m/fused/ 五面板 bzv06_birds_dl_{ctx}.parquet 各 9 成员 (8鸟+DL), 行网格硬 assert 全过。(iii) 三臂 bench 第二轮 LIVE (PID 2519083/2519084/2519085, tmp/bench_r2_arm_{dl,birds_dl,birds}.log, tag bzv06dl_r2_arm_*): @14:42Z arm_dl 至 S_RPE1 (第4/5面板), 结果落 outputs/metrics/local_bench/bzv06dl_r2_arm_{dl,birds_dl,birds}/bizard_summary.json, 对照 round-1 arm_birds 0.24486 / arm_birds_dl 0.1075 / arm_dl -0.894, 门1 需 ≥0.395446; Tier B bench avg_score == 本链 arm_dl (同 best.ckpt delta 源), 无需独立 Tier B 运行。(iv) 官方提交 bizard-v0.5 — scripts/submit_bizard_v05_1016.sh (模板=submit_lam35_1015.sh fixed版, 409-first+窄 allowance, 去 slot-gating) 首发即中: entry 50afg20XylIFiivjWBN9 @2026-10-16T14:20:06Z, model bizard-v0.5, .vcc sha256 0c534e8a 复核一致, @14:43:43Z status=scoring, 额度 10-16 已用 1/2; 2h stuck-launching → vcc cancel --yes + 重提 规则生效。冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励), 10-16额度1/2已用(entry 50afg20XylIFiivjWBN9)
- 2026-10-16T15:40Z [team-lead] (date -u 15:36Z) round-2 三臂终局 + bizard-v0.5 官方结果 + v0.5-vs-v0.4 回归根因裁定 (修正 vintage-swap 假说): (1) round-2 三臂 bench 终读数 — arm_dl mean=-0.8706 (P_H1 -0.8848/Q_HepG2 -0.7160/R_Jurkat -0.6027/S_RPE1 -1.0409/S_HCT116 -1.1084, DL delta 域幅度问题全域复现), arm_birds_dl mean=+0.1207, arm_birds mean=+0.2449, 无臂达门1 (需≥0.395446) → v0.6 不提交走 on_fail=§7 v0.5 保底; DL 负贡献 v0.7 设计输入入档 (raw delta 域需幅度校准/等权 soft_rank 需加权或校准融合/裸 DL delta 不可独立成臂); (2) bizard-v0.5 官方结果 @14:48:55Z published — entry 50afg20XylIFiivjWBN9, Rank 1028, Overall 0.0006606036300101, pds 0.2818/mse 0/nmae 0.0416/fid -0.3629/reach 0.0638/jac -0.0203, Panel vcc2026-val-1, 10-16 额度 1/2 已用; (3) 根因裁定 (sha256 实证, 修正昨段 vintage-swap 假说): 20261016_bizard_v05.vcc ≡ 20261003_bizard_v04_agg.vcc bit-identical (sha256 0c534e8a95b4a08b54c954ab1259ea2ad3d1fccde076424531e6a15804d53a11 双方一致), .h5ad pair 同样 bit-identical (d6ae032c8305d4efcffcdc9dd21beb37c0868b1225205f5c8c3ee5144f61e30e) → v0.5 提交实为 Oct 6 14:11 v04_agg bundle (ref gen2 纯池 8 鸟 soft_rank 等权聚合, local mean5 0.2741, S_HCT116 flattened 0.2688) 原样重传; 官方 v0.4 实为 v02-family λ35 bundle (20261015_bizard_v02_lam35.vcc, script grep F=/L= 实证), 官方 v0.3 = 20261015_bizard_v02_lam37.vcc — v0.5 相对 v0.4 回归不是 gen2-vs-顶层 delta vintage swap, 是 stale bundle copy; (4) 官方计分板全集 (Partition val / Panel vcc2026-val-1): bizard-v0.2/λ47 Rank 908 Overall 0.04565 (10-14, entry BWy91Q4UXdusW6QcxjCU); bizard-v0.3/λ37 Rank 886 Overall 0.05021 (10-15 03:52Z, entry SgXJeIQwv7taWP4B0GND) = 现任最佳; bizard-v0.4/λ35 Rank 887 Overall 0.04912 (10-15 04:22Z, entry 58kzxoM9gGrF1QOdXS5k); bizard-v0.5/坏 bundle Rank 1028 Overall 0.00066; 官方 λ 曲线 (v02-family 内): λ37 0.05021 > λ35 0.04912 > λ47 0.04565 → in-family λ 最优 ~36-38; (5) gen2-pool 家族官方塌方 — local 0.2741 → official 0.00066 (fid -0.044→-0.363, nmae ~0.002/-0.007→+0.0416, pds 稳定 ~0.28), local↔official 分歧本身就是 v0.7 设计输入 (8 鸟等权 soft_rank 聚合的 fid 崩塌与 nmae 恶化指向方向性/幅度校准缺失, pds 稳定说明扰动鉴别结构未丢); (6) v04_results.md §5 A final_c3s (local mean5 0.3455, S_HCT116 scw c3 单鸟 0.6258) 与 B final_pool (0.3214) 从未打包提交 — v04 场景化聚合增益 (scw/heu.25+c3.35, §4 S_HCT116 0.2688→0.5054) 全部留在本地; (7) provenance gap: 20261003_bizard_v04_agg 全 repo 无 build 记录 (scripts/logs/docs/iterations grep 全负), 身份仅可由 v04_results.md §5 "ref gen2 纯池（已打包 20261003）" 推断 — bundle 打包入库纪律缺口 v0.7 制度化 (build_submission 落 manifest: 输入 parquet sha256/λ/聚合方法); 无提交操作, 10-16 额度 1/2 已用; 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)
- 2026-10-16T17:18Z [team-lead] 10-16 第二发 GO: bizard-v0.6 = staged 20261005_dualsrcdeep_lam75.vcc 原样提交 (构建 10-04, sha256 fbdf942806a6802451e3360d10254399c79b76d0bf222a2d946faed60e4c94b5 本段实测复核; 历史: 10-13 E7dtkObkTOI2olKzUEE5 卡 launching 19h → cancel --yes 从未计分, 本 bundle 从未消耗官方计分)。选型依据: (i) 官方记分板 dualsrcdeep 家族全场最优 (λ65=0.0886 Rank633 最佳在档, λ55=0.0872 Rank628, 全部高于 bizard 家族天花板 ~0.05), 且官方 λ 曲线在 λ6.5 斜率仍为正 (fid 七点单调上爬 -0.838→-0.034), λ75 是唯一在档未测的官方 λ 点; (ii) 零构建成本零 vintage 风险 (bundle 10-04 落盘后未动, 规避 v0.5 陈旧重传类事故需在提交后核对 entry sha——本次无法预核因 vcc 不回 sha, 以 F 路径+大小 3,300,679,680 B 为准); (iii) local λ7.5 塌方读数已知 (bench Q nmae -0.551) 且 local↔official λ 偏移有在档先例 (官方峰 λ6.5 vs local 峰 λ5.5), 红队预期 +0.088~0.089 边际改进或持平, 下行风险=λ 阶梯已越顶后回落, 但该信息本身即 v0.7 设计输入 (官方 λ 曲线右端点)。模型名 bizard-v0.6 (序列顺位; 注意 bizard-v0.2 名号曾被 10-05 dualsrcdeep λ65 提交占用过一次, 历史碰撞不阻塞本发)。脚本 scripts/submit_bizard_v06_1016.sh = submit_bizard_v05_1016.sh 固定版模板仅改 F/L/LOG (合规来源, 非 lam37/night_chain), 后台 PID 2553709 启动 @17:15Z, 24 尝试×75s + 180min 轮询, 卡 launching 2h → 人工 cancel --yes 规则待命 (vcc-submission-ops); 上传预计 ~15min (3.3GB @3.5MiB/s), 计分 ~29min。今日额度将 2/2 用满。round-3 融合实验 (加权 soft_rank DL 权重梯度 + DL delta 幅度校准) 与 v04 场景聚合器 port 在本发计分前仅 CPU 侧准备, 不动 GPU/训练; 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)
- 2026-10-16T17:36Z [team-lead] bizard-v0.6 PUBLISHED @17:34Z (entry lrHxrp4sceF58JqMstK3, 提交17:15:18Z, 上传5m25s@9.7MiB/s, scoring~19min, 全程无卡顿): Rank 789, Overall 0.08094471403791086 (pds 0.5184/nmae -0.1548/fid -0.0185/reach 0.1361/jac +0.0044, mse 0, Partition val/Panel vcc2026-val-1, anchor valA/B/C-r4), 10-16额度2/2用满。判读: (i) Overall 较 λ65 (0.0886, Rank633) 回落 -0.0077 — 官方 dualsrcdeep λ 曲线五点终态 λ35/45/55/65/75 = 0.0506/0.0764/0.0872/0.0886/0.0809, 峰值确证 λ6.5, 阶梯在 6.5→7.5 间越顶, 红队预期带 +0.088~0.089 略低于下缘; (ii) 逐项: pds 0.5184 新全档纪录 (>λ65 0.5022), fid -0.0185 champion-class 新纪录 (>λ65 -0.0340), reach 0.1361 纪录, jac +0.0044 纪录 — 方向族四项全部续升; 唯 nmae -0.1548 较 λ65 -0.0656 恶化 -0.089 = λ7.5 幅度过冲打在 nmae 上, 4项增益(+0.0072/6权重)不敌 nmae 一项拖累(-0.0149/6) → Overall 净负; (iii) 公式再证: (pds+mse+nmae+fid+reach+jac)/6 = 0.48566/6 = 0.080944 ✓ (mse=0 项实打实占 1/6 权重拖分, 与 20 快照拟合一致); (iv) v0.7 设计输入强化: 方向族指标与幅度指标在 λ 轴上已分离 (方向仍升/幅度已过), 校准幅度融合 (λ6.5 的 nmae × λ7.5 的 pds/fid/reach/jac) 成为最直接的可乘空间 — 若 nmae 回到 λ65 水平而四项持 λ75, Overall ≈ 0.0957; top-3 门 0.3187 仍需 step-change; (v) local λ7.5 塌方读数 (nmae -0.551) 再次证明夸大 (official -0.1548), local↔official 幅度偏移第三例入档。今日官方提交完毕 (bizard-v0.5 0.00066 坏bundle / bizard-v0.6 0.08094 最佳在档家族新高峰 pds/fid/reach/jac 四纪录), 转入 CPU 侧 round-3: 加权 soft_rank 融合梯度 + DL delta 幅度校准实验 + v04 场景聚合器 port, 全部为 v0.7 候选构建证据链; 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## 2026-10-16 19:26 UTC — round-3 融合实验终局: DL 成员任何剂量均为毒, arm_birds 0.24486 保持本地最优
- w05 崩溃根因: 目录名拼写 w5 vs 脚本传 w05 → bench 回退默认 agg_delta_{ctx} 找不到文件; 修正 --delta-dir …/w5 --delta-pattern 'bzv06_birds_dlw5_{ctx}.parquet' 重跑 (PID 2561330)。
- Plan B 加权 soft_rank 扫描 (Q_HepG2+R_Jurkat, λ2.5): 全线负增益且随 w 单调恶化 — w05 Q 0.2746 (-0.096) / R 0.2965 (-0.074); w10 0.2245/0.2601; w20 0.1773/0.2142; w40 0.1307/0.1767。w→0 即 arm_birds, 5% DL 已拖累 -0.07~-0.10。
- Plan A sd-rescale DL 九成员融合 (5 context 全跑, tag bzv06dl_r3_birds_dlS): mean 0.09791 vs arm_birds 0.24486, 五 context 全负 (-0.044/-0.153/-0.118/-0.168/-0.252); P_H1 -0.0149 / S_RPE1 -0.0339 / S_HCT116 0.0677。
- 裁定: DL delta 对 8 鸟册融合为纯毒 (等权/sd 校准/5-40% 加权全试), 门1 (≥0.395446) 无一接近; DL 臂彻底出局入档 v0.7 设计输入 (方向信息 spearman 0.06~0.22 太弱, 非幅度问题)。arm_birds (8 鸟 soft_rank) 保持 round-1/2/3 本地最优配置。
- 10-16 官方额度 2/2 已用 (v0.5 Rank1028 0.00066 / v0.6 Rank789 0.08094 现任最佳); round-3 结论均为 v0.7 候选证据, 今日无提交操作。
- v0.7 主线维持: 官方 λ 曲线方向/幅度分离 → λ6.5 方向 × 校准幅度融合 bundle 构建 (CPU 侧, 明日额度窗口用); top-3 门 0.3187 需 step-change。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)
- 2026-10-16T18:48Z [team-lead] (date -u) iter022 v0.7 主线定案 + 双探针构建启动: (i) 选型裁定 — dualsrcdeep 家族为 v0.7 唯一主线 (官方在档唯一验证家族, λ 曲线五点 0.0506/0.0764/0.0872/0.0886/0.0809 峰值λ6.5); bz04_final_c3s 出局 (gen2 家族官方塌方先例 local 0.2741→0.00066 未解 + 无已验证 A/B/C submit 网格映射, 预测无意义); final_pool (0.3214) 同族风险备选不上。(ii) v0.7 设计=方向/幅度分离: 官方λ曲线 λ75 方向四项 (pds/fid/reach/jac) 全最优、唯一拖累 nmae -0.1548 幅度过冲; 固定 mf14 帽 (官方唯一验证过的帽) 做纯 λ 轴探针 λ∈{6.5 已测 0.0886, 7.0, 7.5}, 若 nmae 回 λ65 的 -0.0656 而方向四项保持 λ75 水平 → Overall ≈ 0.0957 (+0.0148 vs v0.6 0.08094)。新增 4 配置: dualsrc-deep-lam70-mf14.yaml / dualsrc-deep-lam75-mf14.yaml + bench twin ×2 (逐键镜像), output_stem 20261016_dualsrcdeep_lam70mf14 / 20261016_dualsrcdeep_lam75mf14。(iii) 构建 LIVE: PID 2577437 lam70mf14 @18:45Z 起 (tmp/build_lam70mf14.log), 完成后顺序 lam75mf14 (tmp/run_builds_lam7075.sh, master log tmp/builds_lam7075_master.log); bench waiter PID 2577842 (lam70) / 2577895 (lam75) 待 h5ad 落盘自动跑 Q_HepG2 bench, tag dualdeep_lam70mf14_Q / dualdeep_lam75mf14_Q, 链路=src/eval/local_bench.py (无帽链, 与历史 dualdeep 标签同链可比)。(iv) 三向 bench/submit 预测路径分歧入档: local_bench.py 预测路径无 max_fold 帽 (仅乘 rint) / bench_bird.py build_pred 默认 mf8 / 提交链 make_predictions.py mf12-16 — local 锚不测 max_fold 效应, 故新探针把 mf 显式写进 λ 轴设计 (固定 mf14 解耦 λ-帽共变)。λ/max_fold 阶梯档案: lam45 4.5/12.0, lam55 5.5/12.0, lam65 6.5/14.0, lam75 7.5/16.0。(v) 校准门: bench 落盘后 scripts/official_alignment.py predict --family heur 出官方预测, 应用 +0.012~+0.014 乐观偏置修正 (λ65 实测 pred 0.1027 vs 实际 0.0886; λ75 pred 0.0926 vs 0.0809); 提交判定在双探针读数齐后。(vi) 额度: 10-16 2/2 用满 (v0.5 坏bundle + v0.6 0.08094 现任最佳), 下一窗口 10-17 UTC 00:00, 提交名 bizard-v0.7 顺位, 脚本=submit_bizard_v06_1016.sh 模板仅改 F/L/LOG。冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## 2026-10-16 19:30Z iter022 — overall_pred /5→/6 修正 + 双探针读数定案 + v0.7 bundle 双预打包
- **BUG 修正 scripts/official_alignment.py:162**: overall_pred 曾用 mean(known)=sum/5 (mse 无校准点被静默剔除)， 恒定膨胀 ~+0.017; 官方公式 Overall=(6项)/6 且 mse≡0 → 改为 sum/6。λ65 反演自检: 修后 predict=0.0856 vs 官方实测 0.0886 (偏差 -0.0030)， 校准自洽确证。历史 predict 值一律按 +0.0171 通胀读。
- **双探针诚实预测 (mf14 纯λ轴, /6 修正后)**: lam70mf14_Q → **Overall ≈ 0.0857** (pds 0.5086/nmae -0.0942/fid -0.0416/reach 0.1387/jac 0.0030); lam75mf14_Q → **Overall ≈ 0.0772** (pds 0.5024/nmae -0.1262/fid -0.0530/reach 0.1366/jac 0.0033)。vs v0.6 0.08094: λ70 **+0.005**, λ75 负增益 — "Overall≈0.0957" 乐观场景证伪 (该数字含 /5 通胀)。
- **bench uncapped 确证**: lam75mf14_Q 全六指标 bit-identical 历史未加帽 λ75 — local_bench 链不测 mf 效应; mf14 帽只作用于官方提交 bundle, λ65 校准对 (uncapped-local→mf14-official) 已内嵌帽效应, 预测链成立。
- **build 完工**: lam70mf14 h5ad 4260279969B (validation 2084179052 entries/density 0.5612/integer true); lam75mf14 h5ad 4261125348B (2083946927 entries/density 0.5613/integer true, [done] rc=0 19:15Z)。
- **vcc prep 双双落盘**: lam70mf14.vcc 3300311040B sha256 **019b95422af18ab66b1ea746d34ae3a4660d209cb75e1020ece54ffce69ae5b5** (19:28Z 完成); lam75mf14.vcc 19:30Z 起 prep (PID 2583725)。
- **v0.7 提交裁定 (10-17 UTC 窗口)**: **bizard-v0.7 = 20261016_dualsrcdeep_lam70mf14.vcc** (honest ≈0.0857±0.006, 边际优于 v0.6 0.08094; capped 官方 bundle 可能再优于 uncapped 基预测—λ65 对已含帽效应)。λ75mf14 预测 0.0772 无增益, 留作备件不提交。若 λ70 发回 pds 显著低于 0.5086 或 nmae 劣于 -0.094, 则 λ 轴已到顶, v0.8 转方向项结构性改进 (top-3 门 0.3187 需 step-change, 纯 λ 内插天花板 ≈0.089)。
- 提交脚本: 复制 scripts/submit_bizard_v06_1016.sh 仅改 F/L/LOG → scripts/submit_bizard_v07_1017.sh, UTC 00:00 后启动。
- 额度: 10-16 2/2 用满 (v0.5 0.00066 / v0.6 0.08094 现任最佳); 下窗 10-17 UTC。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## 2026-10-16 19:40Z iter023 — v0.7 λ轴已到顶: 转战 v0.8 方向项结构性改进 (Δ方向头 blend 方案)
- **判定逻辑**: λ70mf14 honest ≈0.0857, λ 轴内插天花板 ≈0.089 (λ65 官方峰 0.0886 + 校准余量) — 即便 λ70 发回达标, 距 top-3 门 0.3187 仍差 3.6×; 纯 λ/max_fold 扫描属参数精修, 需方向项 step-change。四方向项官方 λ65 实测: pds 0.5022/fid -0.0340/reach 0.1267/jac 0.0023 (合计 0.5972/6 = 0.0995 占 Overall 96%: mse≡0 + nmae -0.0656)。
- **方向项结构诊断 (official_alignment_v1.json 11点)**: pds 家族斜率 a=1.667 截距 -0.975 → 官方 pds 对 local 高度敏感 (local 0.85→官方 0.44); fid 官方-0.034~-0.108 全负带 (de_wilcoxon_direction_fidelity_yield_raw: 方向对但 yield 惩罚未达显性 DE 基因集), 提升空间在把非零 delta 集中到 wilcoxon 显性基因上。
- **v0.8 候选方向 (按可执行性排序)**: (a) **Δ方向头 blend**: dualsrcdeep 方向四项已全场最优 + bizard 方向头 (λ47 家族 pds 官方 0.285) → 构造混合 delta (dualsrcdeep 幅度 × bizard 方向符号 agreement gate), 零新训练只组合现成 parquet; (b) Δ top-k 集中化: 把 λ65 delta 按 |lfc| 重排, wilcoxon 显性基因份额 ↑; (c) context-specific λ (A/B/C 各自最优 λ 不共享)。非冻结项, 不动 BizardNet 方向头本体/λ链/affine。
- **行动**: 今晚 CPU 侧先做 (a) 的 agreement 统计: dualsrcdeep λ65 delta vs bizard 方向头 delta 符号一致率 (Q_HepG2 5.6k 基因), 一致率 >0.65 才值得做混合 bundle; 产 tmp/delta_agreement_stats.json。
- 额度 10-16 2/2; bizard-v0.7 (λ70mf14) 明窗照发 — 若 agreement 路线今天就绪可换, 以实测一致率定。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## iter023 续 (2026-10-16 20:25Z): (a) 出局 / (b) top-k 集中化拿到 truth 级证据
- **(a) Δ方向头 blend 裁决出局**: agreement 定稿 dualsrcdeep vs bizard-v02 符号一致率 A 0.6432 / B 0.6256 / C 0.6327 (均值 0.634 未过 0.65 门; |Δ|加权 0.7501-0.7655, tmp/delta_agreement_stats.json); truth 层补刀: bizard 提交头 300 官方目标与本地 bench 100 目标零交集, blend 在本地无 truth 验证路径 → 出局入档。
- **目标集错配教训入档**: data/raw/controls/pert_counts.csv (300 官方评测目标) 与 data/interim/local_bench/Q_HepG2/targets.json (100 bench 代理目标) 交集=0; 首版 truth 探针 bizard 头 0 配对即此因, bench truth 只能验证签名侧头。
- **(b) top-k 集中化 truth 证据 (tmp/delta_heads_vs_truth.json)**: dualsrcdeep 头 (gwps+hct116_all 均权) vs Q_HepG2 truth (99 targets × top-50 |lfc| 基因 vs NTC bulk log2fc, 4695 配对): 全基因 sign 0.6584 (|lfc|加权 0.6959); 头内按 |Δ| top-k 集中后命中率单调上升 — k=500 **0.8553** / k=1000 0.8262 / k=2000 0.8030 → 大 |Δ| 基因方向高度可信, 与 iter023 fid yield 惩罚诊断 (需集中显性基因) 同向。
- **v0.8 裁决**: (b) 立项 — 在 delta 侧按 |Δ| per-target rank 做集中化/收缩 (λ 轴不动), 先 local bench Q_HepG2 验证 pds/fid 响应再定 bundle; (c) context-specific λ 排后。
- 额度 10-16 2/2; bizard-v0.7 (λ70mf14, sha256 019b9542) 10-17 UTC 00:00 窗照发 (scripts/submit_bizard_v07_1017.sh 就绪)。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## iter024 (10-16 UTC 20:30) — (b) top-k A/B 第一轮: bizard gen2 链尾截 nilpotent, 战场转移到 signature 链
- bzv08 A/B (bench_bird λ2.5 Q_HepG2, 变体=per-target |Δ| q25 尾收缩×0.15/置0): overall none 0.4139892 / shrink15 0.4139517 / hardzero 0.4139516 — 三者差 <1e-4, 无效应。
- 机制: bizard gen2 agg_delta 弱尾 |Δ| 均值 0.00046 (q25 以上带 0.0415), 2^2.5×0.00046≈1.0008 折叠 → 对 DE/wilcoxon/任意指标不可见; 该 delta 表天然已集中, 尾截无可移除质量。硬零 25% 行连 nmae 都不动 (0.423851→0.423822)。
- 推论: (b) 的 truth 证据 (top-k 0.8553) 测在 dualsrcdeep signature 链 (|Δ|~O(0.1-2), λ7.0 尾部折叠~1.28 有实质量); 集中化 A/B 必须在该链做。load_delta 冻结 top_k 键 (per-target 全局|Δ| top-K 保留其余置0) 即现成实现, bench twin 配置 bench-dualdeep-lam70-mf14 + top_k∈{500,1000,2000} 直测, 对照=dualdeep_lam70mf14_Q (pds 0.8156/nmae 0.9830/fid 0.5870/reach 0.4217/jac 0.0762)。
- 判据: fid/reach/jac 任一结构性改善且 nmae 不塌 → v0.8 = lam70mf14 + top_k 最优档; 全平 → (b) 在链上也 nilpotent, v0.8 转弃。
- bizard-v0.7 10-17 UTC 00:00 窗照发不变; 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## iter024 续 (10-16 UTC 20:55) — (b) 第二轮 signature 链 top_k A/B: 全指标单调恶化, (b) 裁定出局
- bench twin lam70mf14 × top_k (Q_HepG2, 对照 anchor 无 top_k): k500 pds0.8225(+)/mse0.0263(+)/fid0.5402(-0.047)/reach0.3147(-0.107)/jac0.0633(-0.013)/nmae1.0198(+3.7%差); k1000 fid0.5695/reach0.3409/jac0.0678/nmae1.0073 全劣于 anchor; k2000 fid0.5831/reach0.3776/jac0.0723/nmae0.9927 仍全方向劣。k 越小恶化越重, 单调向 anchor 收敛。
- 结论: 真值链上 per-target |Δ| 尾截 = 砍掉 reach/fid 有用信号换 mse 微利, 方向四项全负 → 与 bizard gen2 链 nilpotent 结论合并: (b) top-k 集中化双链证伪, v0.8 裁定出局。
- 教训对齐: truth top-k sign_agree 0.8553 是"大|Δ|基因方向更可信"的筛后统计, 不等于"砍尾能提分" — reach 需要的是广度不是纯度; wilcoxon DE 集过窄直接塌 fid/reach/jac。
- v0.8 无新立项: (a) blend 已出局, (b) 双链证伪, (c) context-specific λ 依赖 300 官方目标无验证路径且 λ 轴到顶已判。三候选全清 → v0.8 结构项空窗; 主线退守: 10-17 UTC bizard-v0.7 (lam70mf14) 照发, 其官方回数即 v0.8 决策输入。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)
Sat Oct 17 12:43:22 AM UTC 2026

## iter025: bizard-v0.7 官方回数 — 0.0858 published, Rank 782 (2026-10-17 00:43 UTC)

- **提交**: entry 2a8UWn0MwC1l3Xg7ReSJ, model bizard-v0.7, submitted 2026-10-17T00:01:03Z (attempt 11 过闸, 00:00 UTC 重置后首炮; 23:48 UTC 启动脚本撞 409 × 10 次后自动过闸). 00:27:17 UTC published, 全程 26 min 无卡 launching.
- **官方成绩**: Overall **0.08580925**, Rank **782**; 分项 pds 0.51206 / mse 0(官方恒0) / nmae -0.10885 / fid -0.02442 / reach 0.13248 / jac 0.00359. Panel vcc2026-val-1, Partition val, Anchor vcc2026-valA/B/C-r4.
- **与预测对照**: λ70mf14 预测 ≈0.0857 (线性内插天花板 0.089) → 实测 0.08581 命中预测带 (差 +0.0001), λ-shift 内插模型在官方首次**确认**. 上一次 v0.6 同框架实测 (若官方分项可比: 方向四项官方合计 = 0.51206-0.10885-0.02442+0.13248+0.00359 = 0.51487, 占 Overall 96.4% 量级 — 与 iter023 诊断"方向四项占96%"结构一致).
- **Rank 782 与 Top-3 门 0.3187 差距 = 0.2329** — 纯 λ 轴不可能 (到顶已判), 必须方向侧结构性突破. v0.8 三候选已全出清 (iter024), 结构项空窗确认.
- **决策**: v0.9 = 等下一发额度 (10-17 1/2 已用, 剩 1 发) + 官方分项作为唯一新输入; λ 轴不再投入 (λ70mf14 已是链顶). 本轮唯一正收益信息: 内插预测带首次官方验证 ✓.
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)
Sat Oct 17 02:25:00 AM UTC 2026

## iter026: v0.9 深挖 — heur 家族 per-metric 官方标定修复 + 第12点回填 (2026-10-17 02:25 UTC)

- **标定纠错 (关键)**: 预压缩期结论"per-metric 家族拟合不能跨 bench 链迁移到官方分项"是**错族错误** — 当时把 bizard 家族拟合套在 dualsrcdeep(heur) 链的 λ70mf14 anchor 上。本轮直接比对两个 anchor 的 Q_HepG2_scored.csv 分量尺度 (λ65 pds0.8862/nmae-0.3230/fid0.8440/reach0.2251/jac0.0994 vs λ70mf14_Q pds0.8902/nmae-0.4266/fid0.8445/reach0.2496/jac0.1004) → 尺度几乎一致, **heur 家族才是 dualsrcdeep 链的正确映射**。用旧 11 点 heur 拟合预测 λ70mf14: overall 0.08574 vs 官方实测 0.08581, 误差 ~7e-5 — per-metric 官方预测仪器对 dualsrcdeep/heur 链正式**生效**。
- **第12点回填 + refit**: `("heur","20261017_dualsrcdeep_lam70mf14")` 入 OFFICIAL_POINTS (official pds0.5121/nmae-0.1089/fid-0.0244/reach0.1325/jac0.0036), refit 后 heur 系数 (n=4): pds a=1.71698 b=-1.01811 / nmae a=0.28068 b=+0.01613 / fid a=2.66956 b=-2.28962 / reach a=0.61898 b=-0.02072 / jac a=0.61446 b=-0.05842。
- **回代与 LOO**: λ70mf14 in-sample pred 0.08480 (resid -0.0010), λ65 0.08560 (resid -0.0030) — 加点后 overall 级误差 ~1e-3 仍达 gate 判读精度。LOO per-metric: pds 0.0030 / nmae 0.0145 / reach 0.0081 / jac 0.0003 ( tight), fid 0.0457 (max 0.1400 — fid 仍是弱项, 单点杠杆大)。
- **per-metric 灵敏度 (v0.9 ceiling 框架)**: official = a·local + b (heur, n=4)。pds a≈1.72 陡 (local pds=1.0 → official ≤0.70); fid a≈2.67 最陡 (local fid=1.0 → official ≤0.38); nmae a≈0.28 压缩 (local 大摆官方微动); reach a≈0.62; jac a≈0.61。→ **fid 是官方杠杆最大、标定最不稳的指标**: local fid 从 0.84 → 1.0 的理论官方增益 +0.42/6 ≈ +0.070, 单此一项若做实即超剩余差距 0.233 的 30%。
- **v0.9 决策框架 (三重约束下)**: λ 轴到顶 / top-k 双链证伪 / blend+DL 出局。唯一未封死路径 = fid/reach 侧的结构性改进, 且现在**有了提交前验证闭环**: local bench 跑分 → heur per-metric 预测 → overall_pred 过 gate 才花额度。判据: 候选须 overall_pred ≥0.10 (保守 gate, 官方 LOO 误差 ~3e-3 级) 才动用 10-17 剩余 1 发; 否则额度 hold 至 10-18 重置。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)
Sat Oct 17 03:40:00 AM UTC 2026

## iter027: gamma sharpening 双臂 A/B 裁定 — 弃 (2026-10-17 03:40 UTC)

- **验证闭环首次全流程跑通**: var-str fix + eliminate_zeros → cell-eval2 (pdex) → bench_score (bundle 模式) → official_alignment predict --family heur。双臂各 ~2.2 min, 全程 4.5 min。修复 1 坑: pred X 含显式零, numba_mwu `_validate_csr` 拒收 → `X.eliminate_zeros()` 前置。
- **结果 (Q_HepG2, anchor 归一化 per_metric vs anchor raw pds0.8156/nmae0.9830/fid0.5870/reach0.4217/jac0.0762)**:
  - **sharpen_g0.5 (收缩 γ=0.5)**: raw pds0.7635/mse2.650/nmae1.0297/fid0.5383/reach0.3570/jac0.0700 → per_metric overall 0.1621; heur predict overall_pred **-0.036** (fid 官方 -0.3432 塌方) → 远低于 gate, 弃。
  - **sharpen_g1.5 (放大 γ=1.5)**: raw pds0.7623/mse2.150/nmae0.9883/fid0.5715/reach0.4218/jac0.0773 → per_metric overall 0.2400; heur predict overall_pred **0.0239** (pds0.2494/fid-0.1329/reach0.1340) → 低于 gate 0.10, 弃。
- **机制解读**: γ<1 把 pred 向 target mean 收缩 → fid/reach/nmae 全塌 (矩阵整体变平, DE 信号同步衰减); γ>1 恢复至 ≈anchor 水平但无一指标超越 anchor (raw fid 0.5715 < anchor 0.5870, reach 0.4218 ≈ anchor 0.4217 打平, pds 0.7623 < 0.8156) — **单调性确认: γ=1 (anchor) 即该变换族的峰值, 两侧皆劣**。γ≠1 不可救 fid/reach, 与 top-k 证伪同构: pred 分布形态扰动 (收缩/集中/砍尾) 一律伤方向指标, anchor 分布已是该链最优。
- **v0.9 结论更新**: 结构项候选清单再清一项 (sharpen 出局); γ 轴无峰可移。剩余 1 发 10-17 额度继续 hold (无 overall_pred≥0.10 候选), 等 10-18 重置 + 新结构项立项。bench_score 的 multi_ref overall_mean (g0.5: 0.1469±0.0549, g1.5: 0.2274±0.0453) 仅作稳健性参考, 不改变裁定 (heur predict 已含官方标定)。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## iter028: hepg2 自源结构项裁定 — 弃 (falsified-for-validation, 无需实验) (2026-10-17 03:55 UTC)

- **候选**: 把 hepg2_all.parquet (22,989,228 行 / 2,393 targets / 9,622 genes) 加进 dualsrcdeep Δ 链 (datasets: [gwps, hct116_all, hepg2]), 用同细胞系真实 sc 信号补 fid/reach — v0.9 结构项清单上一项。
- **LOCO 兼容性核查 (通过)**: 冻结项 「LOCO 变体禁令维持」 管的是评估方案变体 (置信度加权 LOCO 恢复检验: 从一致性计算留出 RPE1/HepG2; 学习型迁移模型 LOCO train/eval 切分), 不是 Δ 源数据组成 — 加签名源不触禁令。
- **致命缺陷 (in-sample 污染, 判弃)**: configs/local_bench/contexts.yaml Q_HepG2 读 `data/public/scperturb/Nadig2024_hepg2.h5ad`, 而 build_signatures.py DATASETS `"hepg2"` 映射同一文件 — hepg2_all.parquet 就是本地 bench truth 的同实验 pseudobulk。把它放进 twin bench 的 Δ 链 = 让 pred 背诵本地参考自身的 logFC, 本地分 (恰是候选靶向的 fid/reach) 乐观膨胀, 验证闭环 (bench→heur predict→overall_pred≥0.10) 无法合法认证该臂。先例: fusion v1/v2 因 in-sample (训练池含 hepg2/jurkat) 被 「乐观泄漏」 裁定排除出标定拟合 (本地官方校准框架.md:117), 只能单列参考。
- **官方迁移路径也不成立**: 官方 A/B/C context 匿名 (未证实 HepG2-like), hepg2 特异记忆收益无既定迁移通道; 且 heur affine (n=4) 无 in-sample 点, 从污染本地读数外推 = 违反标定族前提。
- **裁定**: 弃 (falsified-for-validation) — 该臂在现行验证框架下不可合法评估, 不建 twin bench config, 不花 bench 算力。与 CCL_2 同类: 有数据价值但无合法评估通道的项不入结构项清单。
- **v0.9 状态**: 结构项清单再度空窗。10-17 剩余 1 发继续 hold (无 overall_pred≥0.10 候选), 等 10-18 UTC 重置。fid/reach 结构性改进仍是唯一未封死路径, 下一轮从新角度立项 (非同源数据 / 评估框架内合法项)。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## iter029: 第三源加源双臂 A/B 裁定 — R1 通过 → bizard-v0.8 GO (2026-10-17 04:30 UTC)

- **候选**: Δ 链第三源扩展 (双源 [gwps,hct116_all] → 三源/四源), 补 fid/reach 方向覆盖。R1=+k562_ess_all, R2=+jurkat_all+rpe1_all; 均对 Q_HepG2 出样 (源 h5ad 与 Nadig2024_hepg2 不同, 无 in-sample 污染; 官方 A/B/C 匿名无泄漏通道)。LOCO 禁令不触 (管评估方案变体非数据源组成, iter028 先例)。
- **闭环全流程**: local_bench (config-only 加源) → bench_score (bundle+multi-ref) → official_alignment predict --family heur (n=4 标定, iter026 生效)。
- **R1 (+k562_ess_all) 全五指标优于 anchor** — anchor 归一化: pds 0.8988 vs 0.8902 (+0.0087) / nmae -0.4237 vs -0.4266 (+0.0029) / fid 0.8673 vs 0.8445 (+0.0227) / reach 0.2848 vs 0.2496 (+0.0351) / jac 0.1051 vs 0.1004 (+0.0047), **avg 0.2764→0.2887 (+0.0123)**。raw: pds0.8186>0.8156 / nmae0.9825<0.9830 / fid0.5965>0.5870 / reach0.4431>0.4217 / jac0.0775>0.0762。**heur overall_pred = 0.1016 ≥ gate 0.10** → 10-17 第二发 GO。
- **R2 (+jurkat_all+rpe1_all) 弃**: pds -0.1566 / nmae -0.4011 collapse (四源均权稀释方向信号, 与 fusion v2 教训同构), overall -0.0420; multi-ref 0.2172±0.0571 也显著弱于 R1 0.2758±0.0440。**加源 ≠ 越多越好, 源的信号质量与主族同向性决定成败** — k562_ess 单源臂通过而双源臂塌方。
- **与既往证伪的结构性区别**: top-k / γ-sharpening / w133 重加权都是**分布形态扰动** (单臂无超越), R1 是**新出样信息注入** (第三 CRISPR 源), 全指标一致改进而非 trade-off 重塑 — 与 fusion v1/v2 乐观泄漏排除的情形也本质不同 (此臂合法出样)。这是 v0.9 以来首个过 gate 的结构项。
- **风险留痕**: 单 seed Δ+0.0123 vs multi-ref σ≈0.0440 (MC 噪声同量级, 有翻车可能); fid 的 heur 标定 LOO 0.0457 弱 (单点杠杆大)。但方向一致性 (五指标同向 + multi-ref 均值同向) 是既往证伪臂不具备的证据等级, gate 判定按规则通过。
- **行动**: 3src 提交 config (dualsrc-deep-lam70-mf14-3src.yaml, anchor 逐键一致仅 datasets) 已建, build+prep 链路启动 (make_predictions.py → vcc prep -g/--perts, 与 iter022 同链), → bizard-v0.8 提交。已提交官方 artifact 永不重算。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## iter029-修正: 3src bundle 字节碰撞 — k562_ess 对官方300靶零覆盖, R1 裁定不迁移 (2026-10-17 05:05 UTC)
- 提交前核查发现 sha256 碰撞: 20261017_dualsrcdeep_lam70mf14_3src.vcc = 019b95422af18ab6... 与 anchor (bizard-v0.7) 逐字节一致 (h5ad 4,260,279,969B / vcc 3,300,311,040B / manifest entries 2,084,179,052 逐项相同)。
- 根因 (load_delta targets 过滤): k562_ess_all 2057 靶全为必需基因筛选谱, 与官方 pert_counts.csv 300 靶交集=0; make_predictions.py `part = g[target_gene.isin(targets)&isin(covered)]` 把 k562_ess 行全数丢弃 → Δ 与 2src 完全相同 (seed/panel 同) → 字节级重复 artifact。
- bench 假阳性机制: local_bench 用 bench targets (2394, k562_ess 覆盖 2055) → R1 五指标增益真实但验证对象不是官方提交对象; bench→submission 靶清单错位 = 验证闭环新缺陷 (同族 in-sample 污染: 验证集合≠评估集合)。
- 处置: vcc cancel 9XItMxvCobntWFW6h5zt --yes (05:00:34Z, status failed "Cancelled by submitter", 槽位已回收) — 10-17 第二发保住; bizard-v0.8 名未发档, 留给下一真实提交。
- 官方300靶覆盖实测: gwps 272 / hct116_all 282 (并集 299, 未覆盖=SPATA20); k562_ess_all / hepg2_all / jurkat_all / rpe1_all / norman_all / tahoe_p12 交集全部=0 → k562_ess 对官方 Δ 链无任何合法注入通道 (generic 兜底仅 gwps, 可影响仅 SPATA20 1/300 靶, 可忽略) → 结构项裁定 falsified-for-submission, 清单再空窗, 10-17 剩 1 发继续 hold 等 10-18 重置。
- 教训固化: 验证闭环必须以官方 300 靶复核签名覆盖行数 (parquet 读 target_gene 列一行即判), bench 通过≠提交有效; 后续任何加源/换源候选先跑 official-300 overlap 检查。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## iter029-修正2: hct116_all_3batch 裁定弃 — 同数据旧代备份, 非独立源 (2026-10-17 05:25 UTC)
- 3batch.parquet 系 build_orion_signatures_v3.py 备份的 hct116_all 旧代 (同 xatlas HCT116_3batches.h5ad, 不同构建参数), 非独立第三源。
- 官方202共同靶 779万对 logFC: abs差中位数 0.0005 (p90 0.0363, corr 0.52) — 近重复; 加入=给 hct116 双倍权重, 分布形态扰动 (与 top-k/γ 同构), 非新信息注入。
- 覆盖零增益: 3batch-only vs hct116_all (official)=[]; SPATA20 仍无解 (gwps∪hct116_all∪3batch=299/300)。
- 裁定: 弃 (同数据冗余源)。通过 coverage law (交集202>0) 但信息量=0, 不建 config 不花算力。v0.8 重复 artifact 已隔离至 tmp/QUARantine_dup/ (outputs/submissions 清空防误提)。
- 结构项清单仍空窗: 现有九签名源中仅 gwps/hct116_all 有官方交集, 全部加源组合已被枚举穷尽 (k562_ess/hepg2/jurkat/rpe1/norman/tahoe 交集0, 3batch 冗余, norman 3靶可忽略) → 加源路径正式封死; 下一结构项需新信息族 (非签名源维度)。10-17 剩 1 发 hold 等 10-18 重置。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## iter029-修正3: 新维度全枚举负面清点 — 聚合方式/置信度链/新数据族均无合法未测候选 (2026-10-17 06:00 UTC)
- **背景**: stop-hook 指令「下一结构项须换维度 — 聚合方式/置信度链/新数据族」。对三维度全部已知候选逐一核对既往裁定, 结论=无一具备合法未测资格, 无 overall_pred≥0.10 候选。
- **聚合方式** (全部有否决记录): top_k (iter024, k500/1000/2000 单调劣于 anchor, k→anchor 单调收敛); w133 dataset_weights {gwps0.7,hct1161.3} (progress line 130 「分布形态扰动, 单臂无超越」; 本次补齐 raw 对比: w133 0.8059/0.9385/0.5810 vs plain 0.8090/0.9235/0.5850, pds/nmae/fid 全劣, 仅 reach +0.0001/jac +0.0012 噪声级微正); γ-sharpening (同 line 130 弃); C3 全局臂权重 (closed FAIL −0.018, cb_v22_c1_freeze.md §7 防复开)。
- **置信度链** (全部有否决记录): L1.7 方向置信度加权 (2026-09-30 confidence_summary.md: 全组合 fid/overall 下降, LOCO 证明效应方向本身为负, 「置信加权 ≈ 基因选择性 λ 缩水」, reach collapse 主导; 结论信号只能用作子集选择/方向先验而非幅度缩放); A.1 ρ-weighting/target-prior v1 (source_lopo_report.rerun.fail.json: gate.pass=false, rpe1 +0.13pp / hepg2 +0.06pp vs 预注册 +2pp AND 双 holdout → 按 rev2 裁定「门过方可接入、门不过自动顺延」target-prior v1 自动顺延); gene_weights (受 A.1 门禁前置约束); sign_fix (闭环冻结不重启)。
- **新数据族**: scbasecount (35,266 样本仅 5 官方靶, 全为 drug/inhibitor/antibody 非 CRISPR KO → 弃); CCL_2 无公开下载链接不可用作训练数据。
- **裁定**: 三维度候选全部携记录在案否决判决或冻结封闭, 本轮无合法未测项, 不花算力不建 config。10-17 剩 1 发继续 hold (无 ≥0.10 候选) 等 10-18 UTC 重置。fid/reach 结构性改进仍是唯一未封死路径; 下一轮若开新项须先过 (a) official-300 coverage law (b) 信息非冗余 (c) 非分布形态扰动三重门。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## iter029-修正4: 结构项审计终局 — 采样密度轴封死, 四维度全清 (2026-10-17 06:08 UTC)
- **背景**: iter029-修正3 封死 聚合方式/置信度链/新数据族 三维度后, 对剩余未裁定 leads (context-specific λ / v04 场景聚合器 port / 采样密度 / max_fold) 逐项终审, 关闭结构项审计。
- **context-specific λ**: 已裁定出局 (iter024续 line 78 原判在档: 依赖 300 官方靶无验证路径 + λ 轴到顶)。本次补证: 代码层确认 make_predictions.py 预测循环对 6 CONTEXTS 用单一全局 cfg["lambda_real"]/cfg["max_fold"], config 无 per-context 键 → 任何落地都需改代码+无验证路径, 双重不可行。
- **v04 场景聚合器 port**: 已裁定出局 (iter022 line 40: bz04_final_c3s gen2 家族官方塌方先例 local 0.2741→0.00066 未解 + 无已验证 A/B/C submit 网格映射; Plan B weighted soft_rank w05→w40 全负单调, DL 方向 spearman 0.06~0.22 太弱, 「DL delta 对 8 鸟册融合为纯毒」line 34)。
- **采样密度轴 (本轮新审, 三项全否)**: (i) entries_per_cell/density_margin 非旋钮 — 仅 validate() 输出读数 (src/models/l0_context_mean/make_predictions.py:119-128), 帽 MAX_TOTAL_ENTRIES=4.75B, 锚 density_margin 0.5611 ≫ 5% 安全线, 无失败模式在档 → 无杠杆; (ii) N_CELLS_PER_PERT=400 硬编码常量 (l0 同文件:27) 非 config 键, 360000=300×400×6 官方 schema 锚定; 400 采样细胞携带的 Δ 完全相同 (factor=2^(λ·dv) 逐 pert 常量), 采样仅注入 Wilcoxon 组内噪声 — 改密度=分布形态扰动 (iter024/iter027 先例: 「anchor 分布已是该链最优」) 且无验证路径 (official-300 ∩ local bench truth = 0), 三重门 (b)信息非冗余/(c)非分布形态扰动 双杀; (iii) 引导子重复加权假说已实测否决 — gwps mult=1.00 (81,162,977 rows/unique pairs), hct116_all mult≈1.0006 (670,122,948/669,741,072), 无静默重复引导子加权 (tmp/mult_check.out)。
- **max_fold 轴**: 维持 line 40/54 原判 — 设计已解耦 (固定 mf14), 纯 λ/max_fold 扫描属参数精修, 不足以覆盖 0.2329 gap。
- **裁定**: 结构项四维度 (签名源加源/聚合方式/置信度链/新数据族) + 参数轴 (λ/max_fold/采样密度) 全部清点完毕, 无一合法未测项, 无 overall_pred≥0.10 候选。10-17 剩 1 发继续 hold 等 10-18 UTC 00:00 重置。fid/reach 结构性改进仍是唯一未封死路径; 若 stop-hook 后续指令点名新维度, 仍须先过 (a) official-300 coverage law (b) 信息非冗余 (c) 非分布形态扰动 三重门。审计闭环, 无 config 无算力消耗。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## iter029-修正5: housekeeping + fid/reach 正交性首条量化 (2026-10-17 06:25 UTC)
- **进程清理**: /tmp/diag_dupes.py (PID 1608661) 运行 55h 未退出 — 七源 (target,gene) 重复诊断, 已被 tmp/mult_check.out 取代且剩余 5 源官方交集全 0 (输出无价值), kill 释放 1 核 + 107GB RSS。
- **stress_alpha 复核**: iter003 (2026-09-30_iter003 L13网格) 已裁「应激正交化无效」— gwps 全局应激签名幅度 |mean logFC|≤0.04 低于 DE 分辨率, α0.3/0.6 与 L1.2 打平, 减不掉东西 — 该 micro-lead 有既有判决, 无需重测。
- **fid/reach 正交性 (新量化, per_target_anchor.csv n=100 靶)**: 靶级 fid↔reach corr 仅 0.15 (nmae↔reach 0.29, fid↔nmae −0.13) — 两指标在靶级别近似独立改进面; anchor fid 四分位 [0.531, 0.593, 0.658], reach 四分位 [0.133, 0.404, 0.634]。含义: fid 结构改进不必然拖累 reach, 但 heur 标定中 fid LOO 最弱 (0.0457), 单靶杠杆大 — 任何 fid 手术须逐靶复核 overall_pred 而非均值外推。
- **裁定**: 审计结论不变 — 四维度+参数轴全清, 无 ≥0.10 候选, 10-17 剩 1 发 hold 等 10-18 UTC 00:00 重置。GPU 空闲 (无 se600m 训练在跑)。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## iter029-修正6: per-target 零 reach 结构量化 — 封顶 +0.0075, 远低于门 (2026-10-17 06:45 UTC)
- **数据源**: outputs/metrics/local_bench/dualdeep_lam70mf14_Q/Q_HepG2_eval/results.csv (cell-eval2 逐靶原始, 100 靶全) — 修正5 的 per_target_anchor.csv 缺失值溯源完成。
- **结构发现**: reach 空值 18 靶全部 jac==0 (预测 DE 集与真值 sig 零交集 = 结构性零分, 聚合时按 0 计入 count=100, 非缺失); 另有 18 靶 nmae 空但 reach 正常 (合计 nmae 空 36, 聚合 count=64)。
- **上限测算**: 18 个零 reach 靶即便全部拉到 reach-OK 中位 0.404, local reach +0.0728 → 官方 reach +0.0450 (a=0.61898) → overall_pred 仅 +0.0075 — **距 0.10 门差 13 倍**, 且无任何已知杠杆能定向修复这 18 个零交集靶 (细胞型特异生物学, 非模型参数可达)。
- **裁定**: per-target 结构手术路径 (含「修复零 reach 靶」假说) 正式关闭 — 逐靶收益天花板远低于门, 不建 config 不花算力。至此 fid/reach 逐靶维度的最后一个开放假说也有量化判决。10-17 剩 1 发继续 hold 等 10-18 UTC 00:00 重置。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## iter029-修正7: 工作区清点闭环 — 全部未跟踪 tmp/ 工件均已裁定, 零悬置 leads (2026-10-17 07:45 UTC)
- **清点动机**: stop-hook 连续质询「循环必须继续」, 对 repo 内所有未跟踪 tmp/ 实验工件逐一定性, 排除「有实验跑完但未入档判决」的悬置状态。
- **结果 (全部已裁定)**: (1) `tmp/bzv08_topk*` / `bench_topk{500,1000,2000}.yaml` / `run_topk_*` = iter024 top-k 集中化 A/B, 双链证伪出局 (reach 需广度非纯度, 砍尾四项全负); (2) `dualdeep3src_k562 / dualdeep4src_jr_lam70mf14_Q` bench dirs + `anchor_normalized_3src.json` (10-17 00:15) = iter029修正 3src 链, 官方300靶零覆盖 byte-identical 判决, bench 读数仅存档; (3) `tmp/mult_check.{out,err}` = iter029修正4 采样密度轴 (mult=1.00) 入档; (4) `delta_heads_vs_truth.json` / `per_target_*.csv` = iter023 truth 证据 + iter029修正6 per-target 封顶, 均已引用。
- **裁定**: 工作区零悬置 — 四维度 (签名源/聚合/置信度链/新数据族) + 参数轴 (λ/mf/密度) + per-target 手术, 每个开过的实验都有在档判决。结构审计全谱系无遗漏确证。10-17 剩 1 发继续 hold, 下一动作点 = 10-18 UTC 00:00 重置后候选重估。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## 10-18 UTC 00:00 重置决策单 (iter029修正7 后固化, 2026-10-17 07:55 UTC 写入, 防崩溃/压缩丢失)
- **重置 tick 决策树** (按序执行, 不重开已封死项):
  1. 核时 `date -u` ≥ 2026-10-18 00:00 → 额度 2/2 可用; 否则继续 hold。
  2. 候选评估唯一入口 = 三重门: (a) official-300 覆盖 (先查后建, 九源已穷尽); (b) 信息非冗余; (c) 非分布形态扰动。任一门不过直接弃, 不建 config。
  3. 现有在档 bundle 全部已提交或已裁定: λ70mf14=v0.7 已发 (0.08581 现任最佳), λ75mf14 预测 0.0772 备件 (提交=负期望, 除非重估推翻), v04_agg/v05 陈旧 bundle 禁重传 (v0.5 先例)。**无过门候选 → 10-18 两发保持未花, 不为花而花**。
  4. 若期间出现新信息 (官方放新数据/CCL_2 公开链接/外部新签名源落盘) → 先跑覆盖一行检查再走决策树。
  5. 结构审计已全封闭 (修正1-7): 签名源/聚合/置信度链/新数据族/λ/mf/采样密度/per-target 手术 — 重置 tick 不得复开, 除非有**新证据** (非旧读数重读)。
- **终点条件**: 若 10-18 全天无过门候选, top-3 (0.3187) 在当前证据下不可达; loop 继续以 30min 巡检维持, 每次重置日执行本决策单后 noop tick 收敛。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## stop-hook 核分复核 (2026-10-17 07:56 UTC): v0.7 官方分项刷新, nmae 劣化入档
- **实测** (vcc status 2a8UWn0MwC1l3Xg7ReSJ, published): Overall 0.08580925 不变, Rank 782→789 (榜单滚动, 非分变), **分项刷新**: pds 0.5121 (iter025 记录 0.5086→续微升), **nmae -0.1089 (较在档 -0.094 劣化 -0.0149)**, fid -0.0244, reach 0.1325, jac 0.0036, mse≡0。
- **判读**: λ70mf14 的 nmae 落在 λ65 (-0.0656) 与 λ75 (-0.1548) 之间单调带内 — 幅度过冲随 λ 单调, 无非单调意外; heur 逐项预测链 (pred 0.08574 vs 实测 0.08581) 再证有效, 分项误差主要在 nmae 项 (预测 -0.094 vs 实测 -0.1089)。**λ 轴正式到顶**: λ70mf14 nmae 实测劣于预测即 λ∈[6.5,7.5] 无更优点, 与决策单第 3 条一致 — λ75mf14 确证负期望, 留库不发。
- 10-17 剩 1 发 hold 不变; 无新候选。冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)

## iter030: 用户指令重开架构轴 — bizard 系列重设计立项 (2026-10-17 08:10 UTC 写入)
- **指令 (用户原话)**: 「我说了重新设计和开发bizard系列模型啊！要大幅改进进入前3，你竭尽全力。该官方测试时最大化利用，然后本地不断迭代测试优化！」 — 修订 v0.9 门槛 (overall_pred≥0.10 才花): 花发由用户直接授权, 但候选仍须 bench 超 anchor + heur overall_pred 证据才花, 不裸花。
- **解冻入档 (architecture unfreeze)**: 学习式迁移校准器 (learned cross-context calibrator) 为新开放轴 — LOCO 禁令维持其原文范围 (管评估方案变体, line 117 iter028), 学习式模型训练/验证走 held-out-pert 诚实切分。hepg2 自源陷阱 (iter028: hepg2_all.parquet 即 Q bench truth 文件) → 任何含 hepg2 的训练臂一律 hold out bench-100 靶 (train 2290 / eval 100)。
- **证伪教训入档**: bzv06dl DL 方向头纯毒 (arm_birds Q 0.3710 → +DL 0.2469–0.2512; DL-only −0.7654; w-scan 单调负) — 重设计**不从 DL 方向头续建**; v10_full_redesign.md 的端到端 BizardNet 计划 (10-08) 其前提 (方向 spearman 0.06~0.22) 已被后续 bench 证伪, 不复活。唯一有正证据的学习式先例: 校准框架 line 93 — S_HCT116 fusion v2@1.5 fid 0.777 vs 启发式 0.672 (+0.106 ≈ 官方 +0.119), 学习式 fid 修复跨上下文稳健且非 in-sample。
- **跨上下文迁移探针 (task bn0l9qc37, 17,681,299 matched pairs)**: gwps(K562)→hepg2_all(HepG2) 签名 corr 0.2142; per-target spearman median (n=2390) 0.0976; direct transfer MAE 0.0658 vs zero-baseline 0.0577 — **直接迁移劣于零基线, 原样拼接有害, 必须学习校准**。matched pool = gwps∩hepg2_all = 2390 perts; pool∩official300 = 0 (无官方评估污染); pool∩bench100 = 100 (bench 全在 pool 内 → 必须留出)。
- **fid 杠杆**: heur 标定 n=4 (iter026) fid a≈2.66956 最陡且 LOO 最弱 (0.0457) — 官方 fid −0.0244 是最大单项失分; local fid 0.84→1.0 ≈ 官方 +0.070 ≈ 剩余 gap 0.2329 的 30%。fid↔reach 靶级 corr 0.15 (修正5) — fid 手术不必然拖累 reach。
- **第一臂 bzr1 设计**: 学习式 per-gene 源组合/残差校准器 — 训练数据 = 6 上下文两两配对的 matched-pert 签名 (输入=源签名 logfc + 基因级协变量, 标签=目标上下文 logfc), 推理时作用于 Δ 链源签名 (gwps/hct116_all) 生成上下文自适应校正项; 验证 = Q_HepG2 bench-100 held-out + S_HCT116 复核双场景, 判据 = fid 结构改善且 nmae 不塌, 全程走闭环 (local_bench → bench_score → official_alignment predict → overall_pred)。
- **额度**: 10-17 剩 1 发 + 10-18 2 发全部留给过门重设计臂 (不为花而花, 但指令授权积极构造/验证候选, 优先 fid 轴)。v0.8 名未发档留给第一支过门重设计臂。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令(评估方案变体)/PBT奖励) — **解冻: 架构学习轴 (学习式迁移校准器, 用户指令 iter030, held-out-pert 诚实验证强制)**

## iter030-修正1: bzr1 M_ctx v1 诚实留出判决 — 三门全败 FAIL, 残差学习器本身为真 (2026-10-17 09:00 UTC)
- **执行**: `src/models/bizard/bzr1/train_calibrator.py` draft-3 完整跑通 (int64 key join 替代 pandas merge, 去重 bincount mean; 全 vocab 39,084 基因 × 18,301 perts; 训练 85,114,067 对 / hold 3,740,985 对; XGB reg:pseudohubererror hist+cuda 500 rounds, L20 ~15 min/迭代 — 迭代便宜)。
- **诚实验证**: Q_HepG2 bench-100 靶全场景留出 (hepg2_all 侧统计同剔)。基线与探针偏差入档: hold 子集 direct corr **0.0007** (探针全池 0.2142) / spearman med 0.0002 / direct mae 0.0782 vs zero 0.0578 — **bench-100 靶即跨上下文迁移失败的抗性子集** (hold 直接迁移 ≈ 零相关, 全池正)。探针全池读数只能作全池参考, 不能当 hold 预期。
- **模型读数 (hold, 100 靶)**: pred = x + r̂: mae 0.0582 / corr 0.0001 / spearman med 0.0001; **残差学习器诊断 r̂ vs r: corr 0.5863 / spearman med 0.6043 — 模型学到的是真残差结构**。诊断分离 = 失败机理是 pred = x + r̂ 中 x 方差主导且与 y 不相关, 加上 r̂ 反而加了噪声。
- **GATES (预注册)**: corr_beats_direct False / mae_beats_zero False (0.0582 > 0.0578) / spearman_beats_direct False → **VERDICT: FAIL**。v1 不具备花发资格, 未花。
- **v2 假设 (按证据强度排序)**: (a) 直接预测 y (目标改 y 非 residual) — residual 诊断已证模型有学习信号, 让它直接拟合 y; (b) residual shrinkage pred = x + γ·r̂, γ 逐基因/逐扰动学习; (c) 门代理再审视 — Δ 链下游是相对幅度/bench 效果, 与绝对 logfc corr 的门是否对齐需在 (a)(b) 出结果后一并诚实复核。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令(评估方案变体)/PBT奖励) — 解冻仅限架构学习轴 (held-out-pert 诚实验证强制)

## iter030-修正2: bzr1 v2 (direct-y + 目标侧基因先验) 技术 PASS + corr 矛盾追查 (2026-10-17 12:30 UTC)
- **执行**: `train_calibrator_v2.py` — 弃 pred=x+r̂, 直接拟合 y; 14 特征 = v1 10 列 + (tg_mean/tg_std/tg_q90 目标侧 hepg2 train-pert 基因先验) + src_id。同 hold 协议, XGB 500 rounds cuda t≈771s。
- **初报读数**: MODEL mae 0.0527 / corr 0.5104 / spMed 0.2383, 三门 True → 技术性 PASS。但 **direct 基线读数 corr 0.2524 与修正1的 0.0007 矛盾**, 且 v1 FAIL 判决建立在同一基线上 — 判决链存疑, 发花冻结。
- **追查**: reconcile_probe.py (undeduped pandas string-merge 金标准) 独立复算 direct corr **0.2142** (全池, 零重复行) — 与探针 bn0l9qc37 (0.2142) 一致, 两独立路径吻合 → 唯一异常源锁定在 int64 key 管线本身。

## iter030-修正3: 对齐 bug 根因 (np.unique 键排序撕裂值) — v1 FAIL 判决作废, v2 转 REAL PASS (2026-10-17 13:30 UTC)
- **根因** (`sortedness_probe.py` 决定性): `load_keys` 中 `np.unique(keys)` 排序键后, 值重排仅在 `if len(keys_u) != len(keys)` (有重复行) 时执行。实测 gwps 81,162,977 行 / hepg2_all 22,989,288 行 **均零重复行** 且文件序非 key 单调 → 去重分支恒跳过, 键已排序而值留文件序 → (x,y) 对完全错配。BUGGY 路径 corr **0.0003**, FIXED 路径 (值经 inv 跟随重排) corr **0.2142 = 金标准** — 探针与 reconcile 全部平反, 异常 100% 在 key 管线。
- **修复**: train_calibrator_v2.py / diag_per_source.py / train_calibrator.py (v1) load_keys 无条件 bincount-mean 重排。
- **v1 判决作废**: v1 训练数据本身错配 (x,y), 且其门对比的 direct 基线 (0.0007) 也是错配产物 — 双重污染, "pred=x+r̂ FAIL" 结论不得复用。
- **诚实重推 (v2 fixed, hold 100 靶)**: baseline direct mae 0.0701 / corr **0.2524** / spMed 0.0917; zero mae 0.0575; gene-prior (TGM) mae 0.0555 / corr 0.3703 / spMed 0.2375; **MODEL mae 0.0527 / corr 0.5104 / spMed 0.2383** — 三门 True 转**真 PASS** (corr 2.0× direct, 1.4× gene-prior; mae 三者最优; spMed 2.6× direct)。
- **per-source 直接迁移基线 (diag_per_source.py, 修复后, HOLD|POOL)**: gwps corr 0.2210|0.2138; hct116_all 0.2235|0.2136; jurkat_all 0.1622|0.1861; rpe1_all 0.3710|0.3471; k562_ess_all 0.2629|0.2551 — 五源全部携带真迁移信号, α* 0.19–0.42, HOLD≈POOL (无 hold 子集伪影)。
- **遗留注记**: (i) train_meta_v2.json "n_train" 字段误记 hold 大小 (3,740,985), 真训练量 85,114,067; (ii) v1 文件曾被 cp 覆盖为 v2 内容, 已自 b509ccf 恢复并单独打修正3补丁 — v1 真实内容(残差公式)在 git 中, 判决已作废; (iii) 官方 v0.7 链 make_predictions.py 无 intersect1d/load_keys, **不受此 bug 影响**。
- **下一步 (v0.8 路径)**: 校准器 Δ 链落地 — 用 M_ctx_v2 校准 gwps/hct116_all 源签名生成改进 Δ → 候选 bundle → local_bench → heur overall_pred; fid 为官方最大杠杆 (本地 fid 0.84→1.0 ≈ 官方 +0.070)。

## iter030-修正4: v0.8 纯校准签名 bench 判决 — pds 方向崩, heur -0.0218 FAIL 未花 (2026-10-17 21:00 UTC)
- **执行**: make_delta_v2.py (M_ctx_v2 校准器全 vocab 推理, 3 源 vocab 并集, bench-100 hold 硬断言, 修2 对齐律无条件 bincount-mean) → gwps_cal/hct116_all_cal 签名 → bench-bzr1v08-lam70-mf14.yaml → local_bench (Q_HepG2) → bench_score → official_alignment predict (family heur) 全链 rc=0。
- **raw bench (Q_HepG2)**: pds_cosine **0.5520** (v0.7=0.8902, 崩 -0.34) / mse 0.0212 / nmae 0.3060 (改善) / fid 0.9466 (v0.7≈0.84) / reach 0.4714 / jac 0.1608。anchor overall 0.3371; anchors: pds 0.1378 / fid 0.9466 / nmae 0.3060 / reach 0.4714 / jac 0.1608。
- **heur overall_pred = -0.0218** (global n=12 map +0.0023) — **< 0.0858 门 → FAIL, 未花**。
- **根因 (pds 机理)**: pds_cosine 尺度不变 → raw 分数下降即真方向退化, 非幅度伪影。校准器输出整体收缩 (gwps yhat q99 0.048 vs raw x q99 0.126; corr(x,yhat) 仅 0.403/0.374), 把 raw 大基因质量压向基因先验 → top-gene 序重构 → 方向崩。fid/nmae/reach/jac 四项全改善 (fid 0.9466 已近 1.0) — 校准幅度信息为真, 但与方向信息不可同时保留 (线性 blend 几何下)。
- **教训入档**: held-out corr 0.51 的校准器 ≠ bench 方向可保留; 幅度校准与方向保真的冲突是结构性的 (calibrator 以基因先验为锚, 而靶特异方向结构活在 raw 大基因里)。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励) — 解冻仅架构学习轴

## iter030-修正5: v0.9 α-blend 臂 a70 判决 — heur 0.0634 FAIL; α 轴封顶于 α=1≡v0.7, 线性混合轴关闭 (2026-10-17 21:00 UTC)
- **设计**: blend_cal.py 签名级线性混合 y = α·x_raw + (1-α)·yhat_cal; α=1≡v0.7, α=0≡v0.8; pyarrow dictionary-encode 键对齐 (iter030-修正3 同式), 一次产出 calA50/calA70 两套。
- **a70 (=0.7) 全链 rc=0 (20:43 UTC)**: raw pds 0.7048 (v0.7 0.8902 / v0.8 0.5520 — 部分回收) / fid 0.6132 / nmae 0.9062 / reach 0.5203 / jac 0.0818; anchor 0.3447; anchors pds 0.5741 / fid 0.9068 / nmae 0.056 / reach 0.411 / jac 0.1204; **heur overall_pred 0.0634 < 0.0858 → FAIL** (较 v0.8 -0.0218 大幅好转但仍不过门)。
- **α 轴插值封顶 (决策关键)**: 锚点 {α=0: -0.130, α=0.7: +0.380, α=1: +0.509} (sum-of-preds, α=1 即 v0.7 自身 pred 0.0848≈official 0.0858)。pds 预测每 α 单位 +0.73 (0→0.7) 后 +0.45 (0.7→1.0), 边际收益全程为正 → **α 轴极大值 ≈ α=1 = v0.7 本身; fid/nmae/reach 锚随 α 单调下降 (校准只在 α=0 端改善)**。结论: 线性混合轴作为参数化已关闭 — 幅度修正与方向保真在此几何下不可兼得, α→1 即放弃校准收益。
- **a50 未跑** (预算 ~0.03, 插值劣于 a70, 建链 tmp/run_bench_v09_a50.sh 留存不启动; 轴已插值关闭, 补点无判决价值)。
- **下一步 (v0.9b 符号门控)**: 冲突根源在校准器动方向 → 门控只在 sign(x)≠sign(yhat) 的分歧基因上收缩 (sign 一致保 raw 全幅)。分歧率/|x|质量敞口诊断随构建落盘; SG50/SG100 两点 bench 待链。
- 冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励) — 解冻仅架构学习轴

## iter030-修正6: v0.9b 符号门控 + v0.9c 幅度门控 — 四臂全败, signature 手术家族正式关闭 (2026-10-17)

四点失败表 (heur overall_pred, 门=0.0858=v0.7 官方):

| 臂 | 构造 | local pds | overall_pred | 判决 |
|---|---|---|---|---|
| v0.9b-sg100 | 分歧基因全清零 (β=1) | 0.5306 | **0.0284** | FAIL |
| v0.9c-mg90 | 仅 top-10% \|x\| 保 raw | 0.3971 | **-0.0064** | FAIL (全家族最差) |
| v0.9c-mg75 | 仅 top-25% \|x\| 保 raw | 0.5410 | **0.0184** | FAIL |
| v0.9b-sg50 | 分歧基因减半 (β=0.5) | 0.6782 | **0.0543** | FAIL |

参照: a70 0.0634 / v0.8 -0.0218 / v0.7 0.0848(门). pds 阶梯: v0.7 0.8902 → a70 0.7048 → sg50 0.6782 → mg75 0.5410 → v0.8 0.5520 → sg100 0.5306 → mg90 0.3971.

**两个假设同日证伪**:
- 符号门控 (分歧=噪声): sg100 清掉 32%/37% |x| 质量的分歧基因, pds 反而低于 a70 半混合 → 分歧基因携带真方向信号, 清除=破坏结构.
- 幅度门控 (大|x|=载体): mg90 只豁免 top-10% 比 v0.8 纯校准还差 → 中等 |x| 带也载方向. MG 诊断: gwps 分歧 |x| 质量散布 q50/75/90/95 带 = 0.076/0.096/0.080/0.031, hct116 = 0.002/0.050/0.126/0.079 — 分歧载体**不可按幅度分域**.

**家族关闭判决**: 校准后 signature 级手术 (α-blend / sign-gate / mag-gate) 五个干预点全部低于门, 排序≈干预广度; fid 锚 (0.88-0.95 vs v0.7 0.84) 永远改善但补不回 pds. 最大值在 α=1 ≡ v0.7. 结论: 任何在"校准输出→替换/混合原值"范式内的后续臂均为负期望, **不再构造**. 真重设计必须换构造 (方向保持by-design: 例如直接在 fold/2^Δ 空间学习、或以 pds 对齐为目标函数), 否则 10-18 两发按重置决策单执行 (无过门候选→保持未花).

## iter035-续: se600m r2 三臂第二轮 bench 全败 + v0.5 官方已发布补记 + dataset_weights 探针噪声级 (2026-10-18 13:35 UTC)

- **r2 三臂第二轮 (bizard_summary mean_overall, 门1 ≥0.395446 = round-1 v0.6-dl 官方还原)**: arm_birds **0.2449** (与 round-1 逐位一致 0.24486 — 确定性复现, 无种子漂移) / arm_birds_dl **0.1207** / arm_dl **-0.8706** — 三臂全 FAIL, 排序与 round-1 完全一致, 无新信息。链路已停 (无运行中进程, bench log 尾部均为终值)。
- **v0.5 官方结果补记 (从 logs/submit_bizard_v05_1016.log FINAL_STATUS 恢复)**: 2026-10-16T14:48:55Z published, **Overall 0.0006606 / Rank 1028** (pds 0.28178 / mse 0 / nmae 0.04160 / fid -0.36289 / reach 0.06381 / jac -0.02035)。提交~29min 即发布, 卡死2h cancel 规则未触发; entry 50afg20XylIFiivjWBN9 现已 not_found (vcc status), log 为唯一记录。结论: v0.5 系保底提交已处置完毕, 不重算不重提。
- **dataset_weights 探针判决 (主链 bzr1, bench-in-loop 证据)**: gwps/hct116_calNC 重叠 71.0M 对, sign agree 0.5077 (≈抛硬币); 强对 (max|v|>0.25) 仅 0.23% 且 agree 0.6495。Q_HepG2 truth top50 覆盖: 99/99 靶在 gwps, 86/99 在 hct (13 靶 gwps-only 不受权重影响, 0 靶 hct-only)。对 2703 truth 覆盖对扫描 w_g/w_h 比率: 最大 sign-match 增益 +0.0014 (w_g=1.2: 0.7114 vs 等权 0.7100), 净行分解 +26/-22 (净 4 行), 强真值行 (|tru|>0.5, n=1940) 净 0, 增益行 |s_eq| 0.0015 全在噪声带。**判决: dataset_weights 非对称加权非真实结构杠杆, 不建臂**。
- **冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)** — 主线下一结构臂候选: NC+yh 混合 (DE-gene yh 幅度 + NC sign lock, 须守 A70 教训: 大幅混入 cosine 主导签名崩 pds); dataset_weights 已探明噪声级关闭。

## iter035-续2: HY k 曲线完备 — k*=6, HY6 heur 0.0907 全场新高, 首族过 0.0859 门 (2026-10-18 17:50 UTC)

- **k 曲线终态 (bench-in-loop, Q_HepG2, heur overall_pred, 门=0.0859)**: **k=6 → 0.0907 (NEW BEST)** > k=12 → 0.0905 > k=18 → 0.0891 ≈ k=50 → 0.0890 > k=25 → 0.0887 > v10hcx 0.0896 (官方 0.0859) > v11nc 0.0859 (门). plateau 在 k∈[6,12], 之后劣化; k*=6.
- **HYf (只升不缩 max(|x|,|yhat|)) 括号对照 0.0786 FAIL** — 放大再次为负 (AMP 家族判决第四次复现), HY 的 pds 增益来自**收缩**让位而非放大.
- **heur 排序终局**: HY6 0.0907 > HY12 0.0905 > v10hcx 0.0896 > HY18 0.0891 > HY50 0.0890 > HY25 0.0887 > v11nc 0.0859 > S08 0.0844 > HYf 0.0786 ≈ G115 0.0782 > a70 0.0634 >> ampq75 -0.0382.
- **local official-mode anchor**: HY12 0.3144 > HY6 0.3124 > HY50 0.3122 > HY18 0.3115 > HY25 0.3110 > S08 0.3081 > NC 0.2904 > HCX 0.2751 — HY 族在官方口径下同样全场领先.
- **构造要点复述**: 基底 NC(x)=x−μ_g+sign 非穿越翻回; DE 集 = per (source,target) top-k by |yhat| (校准器自身置信, 不触 truth); HY = DE∩sign(x)==sign(yhat) 处 v=sign(x)·|yhat|, 其余=NC(x). HY6 de&agree 覆盖率仅 0.13% (gwps) / 0.01% (hct116) — 结构外科级局部干预.
- **heurs→official 缝隙警告**: v10hcx heur 0.0896 → 官方 0.085867 (gap −0.0037). HY6 线性外推官方 ≈0.0870 (+0.0011 vs v0.8), 薄. 10-18 剩 1 发; 决策 = HY12 (official-mode anchor 0.3144 全场最佳) 作为 bizard-v0.9 提交臂 — anchor 域 (0.3124 vs 0.3144, HY12>+0.002) 与 fid 代理 (HY12 -0.058 vs HY6 -0.0599) 双指标均指向 12, 且 k=12 处曲线平台中心更稳.
- **提交链已启动 (17:52 UTC)**: production config configs/l1_signature/bzr1v12-hy12-lam70-mf14.yaml (datasets [gwps_calHY12, hct116_all_calHY12], output_stem 20261018_bizard_v09_hy12) → make_predictions.py 后台构建 h5ad (task b97pgz6rn) → vcc prep → submit (模板 scripts/submit_bizard_v08_1017.sh, L=bizard-v0.9).
- **冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)**

## 2026-10-18 18:51 UTC — iter035-续3: bizard-v0.9 官方 PUBLISHED — 0.0924 rank 762 新最高, HY 臂官方兑现

- **官方结果 (entry CV9sGkgWRLsWCPlPmqve, published 18:51 UTC)**: **Overall 0.092378** — rank **762**/791, 超越 v0.8 (0.085867, rank 788) +0.0065, 系列新纪录 (v0.7 0.085809 / v0.6 0.08094 / v0.5 0.00066).
- **分项**: pds **0.5062** (v0.8 0.4833, +0.023 — NC+yh 收缩让位结构对 pds 的增益官方兑现) / nmae −0.0575 (HCX −0.058 持平) / fid −0.0288 (HCX −0.0270 持平) / reach 0.1310 / jac 0.0034 / mse 0.
- **heur→official 缝隙实测**: HY12 heur 0.0905 → 官方 0.0924 (gap **+0.0019**, 方向反转为正!) — v10hcx 的 −0.0037 负缝隙不迁移到 HY 族; HY 官方模式 anchor 0.3144 的排序与官方结果一致.
- **判决**: NC+yh 混合臂 (NC 基底 + top-k by |yhat| DE∩sign-agree 让位) 构造类首战告捷, k 轴确认有效; HYf 负括号 (0.0786) 已证负 → 增益 = 收缩让位, 非放大.
- **预算状态**: 10-18 两发用尽 (v0.6 λ75 0.08094 + v0.9 HY12 0.0924); 10-19 00:00 UTC 重置 2 发.
- **下一轮候选排序 (过门后才花)**: 需本地新证据 — (a) HY k∈{3,8} 细化括号 (k=6 是边缘值, k∈[6,12] 平台内有更优可能); (b) NC+yh 构造与其他已关闭轴的组合 (仅 anchor 证据支持时); 无过门候选 → 不为花而花.
- **冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)**

## 2026-10-18 19:21 UTC — iter035-续4: HY k 细化 (k=3,8) 完备 — HY8 双指标新 k*, 准备 bizard-v0.10

- **k 曲线补全 (heur overall_pred / official-mode anchor / bench overall)**: k=3 → 0.0911 / 0.3121; k=6 → 0.0907 / 0.3124; **k=8 → 0.0915 / 0.3154 (双指标新 k\*)**; k=12 → 0.0905 / 0.3144 (官方已兑 0.0924); k=18 → 0.0891; k=25 → 0.0887; k=50 → 0.0890.
- **HY3 证负**: heur 0.0911 但 anchor 0.3121 最低档, bench pds 0.9329 虽高, jac/nmae 恶化 — 过稀 DE 集信息不足; 排除.
- **HY8 判定**: heur +0.0010 / anchor +0.0010 双超 HY12, 与 HY12>HY6 的同款双指标逻辑一致 (该逻辑已被 v0.9 官方 0.0924 验证); 预测官方 ≈ 0.0933–0.0935.
- **预算状态**: 10-19 00:00 UTC (约 4.6h 后) 重置 2 发. 决策 = **bizard-v0.10 = HY8** (第一发), 第二发留待 HY8 结果后依新证据决策.
- **生产链**: config bzr1v13-hy8-lam70-mf14.yaml (镜像 hy12 生产配置, datasets *_calHY8) → make_predictions.py 后台 → vcc prep → submit 脚本带 00:00 UTC 时间闸.
- **冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)**
- [10-18 22:02 UTC] sim10 wrong-drop细分证负: disagree仓位仅0.71/8, 清错号只+pds 0.0006; HYf实测(负)反证sim9放大方向→幅度谱轴全闭合, HY8=该构造类最优点. wrapper 1265702 alive, v0.10=HY8 待00:02 UTC.

## 2026-10-18 22:45 UTC — iter035-续6: RS 幅度恢复臂 bench 证负 — 幅度轴闭合, L269 重训为唯一剩余出口

- **RS 臂已构建+bench (零提交)**: per-target s_t = ||NC_t||₂/||yc_t||₂, v_rs = s_t·yc_t (方向=校准器 by-design, 幅度恢复到 NC 范数). 构建诊断: gwps targets=9,866 s_med=2.181 (q10 1.700/q90 2.946), corr(x,RS)=0.405, q99|RS|=0.170, max|RS|=2.473; hct116 targets=17,358 s_med=1.539, corr(x,RS)=0.372, q99|RS|=0.064, max|RS|=1.590. bench coverage 100.
- **bench 六指标**: pds 0.5480 / nmae 1.0280 / fid 0.6427 / reach 0.6232 / jac 0.0941 / mse 0.02972. heur overall_pred **0.0577** (pds -0.1404/jac 0.0447/nmae -0.1828/fid 0.2792/reach 0.3456; local_anchor 0.1894) — **FAIL 门 0.0859, 零提交**.
- **判决**: RS pds 0.5480 ≈ pure-calibrated-signature 0.5520 — 幅度恢复**未**回收 pds; nmae 反而劣化 (1.0280 vs pure-cal 0.3060 / HY8 0.9456), 仅 fid/reach/jac 较 HY8 改善. bench 崩是**结构性** (校准方向/排序经 λ=7 × NTC 400-cell 采样均值后的管道响应), 非幅度性. raw-cosine probe (YH 0.3588 > NC 0.1572) 第三次未穿过 bench 链 — 三重代理迁移失败律再现, 唯一可信评估器=bench 链.
- **幅度恢复轴关闭**. v0.11 候选菜单终局: Δ-chain 修正4 / α-blend 修正5 / sign-mag-gate 修正6 / RS 本轮 — 手术族全禁 (line 269); v04 聚合器移植排除; DL delta 纯毒. **唯一剩余合法出口 = L269 重训指令: 「直接在 fold/2^Δ 空间学习、或以 pds 对齐为目标函数」的方向保持 by-design 重训** — infra 可自 train_bird.py/train_calibrator_v2.py 适配, benchable 经 bench_bird.py (Δ̂ 臂) 或 local_bench.py (signature 臂).
- 提交链: wrapper 1265702 alive (etime 02:24), v0.10=HY8 待 00:02 UTC 自动发. GPU 空闲 (159 MiB).
- **冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)**

## 2026-10-19 00:35 UTC — iter036: v0.10=HY8 官方兑分 0.092219 (rank 764) — k 微调轴官方关闭, HY 类平台 ≈0.0923

- **官方结果**: bizard-v0.10 (HY8) Overall **0.092219** rank 764 (entry A8L3mpuxVO9GsqtI2JxM, 00:03:06 UTC 提交, 00:31 UTC published). 分项: pds 0.5083 (v0.9 0.5062, +0.002) / nmae −0.0605 (v0.9 −0.0575, 改善) / fid −0.0289 持平 / reach 0.1310 持平 / jac 0.0034 / mse 0.
- **heur→official 缝隙再校准**: HY8 heur 0.0915 → 官方 0.0922 (+0.0007); HY12 heur 0.0905 → 0.0924 (+0.0019). 预测 0.0933–0.0941 高估 ~0.0012 — 双指标 (heur+anchor) k 细化逻辑本次未迁移: HY8 官方 ≤ HY12 (−0.0002). 与 v0.9 的 +0.0019 正缝隙相比, k=8 的 +0.0010 heur 增量被官方噪声吞没.
- **判决**: k∈[6,12] 为官方平台 ≈0.0922–0.0924, 排序抖动 <0.0005; **HY k 微调轴正式关闭** (官方最优 = v0.9 HY12 0.092378, 构造类最优). 后续候选需 >0.0924 期望才值得花.
- **10-19 预算**: 第 1 发已花 (v0.10); 第 2 发候选 = V3(K4,β0.5) (probe_v4 唯一 two-win 点, pds 0.8994>NC 0.7935, sp 0.0917≥0.0842) — 走 bench 门, heur > 0.0859+margin 才花; 已知风险: 校准方向类 bench 三连崩 (pure-cal 0.5520/RS 0.5480 pds), K4_b0.5 probe 读数按三重代理迁移失败律不可信, bench 链才是仲裁.
- **冻结未动(BizardNet方向头/λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)**

## 2026-10-19 01:40 UTC — iter037: V3(K4,β0.5) bench 灾难证负 (overall −0.323) — 校准方向构造族正式封轴

**事实链** (本地 21:10–21:37 EDT = 01:10–01:37 UTC):
1. `build_v3.py` 双源落盘完成 (gwps 81.16M 行 / hct116 669.74M 行, 全 NG=39084 宽度, K=4 β=0.5, eigh≈SVD 等价实现, probe 约定逐项核对).
2. bench 链 (bzr1v16, Q_HepG2): **overall_pred = −0.323**, local_raw pds **0.1367**, nmae −0.4807, anchor −0.0645.
3. 对比同族: pure-cal pds 0.5520 / RS 0.5480 → V3 0.1367 是最差 — 去相关方向变换在 make_predictions 管道里被放大摧毁, probe 全 NG 宽度读数 (pds 0.8994, sp 0.0917 two-win) 完全不迁移且反向.

**裁决**:
- **校准输出→方向变换构造族 (pure-cal / RS / V3) 全部 bench 结构性崩, 轴正式关闭.** L269 的「换构造=方向保持去相关」出口在 bench 仲裁下证伪. 三重代理迁移失败律第 4 次确认, 且证明连零训练全宽 probe 也不可信 — 唯一可信评估器只有 bench 链, 而 bench 对该族给出 0.13–0.55 的死区.
- **10-19 第 2 发不出手**: V3 距门 −0.4, 无任何候选期望 >0.0924. 不为花而花.
- k 微调轴 (iter036) 已关闭; HY 构造类官方平台 ≈0.0922–0.0924; 官方最佳仍 bizard-v0.9 (HY12) 0.092378 rank 762.

**剩余出路评估** (签名-校准范式内全部主轴已闭):
- 架构学习轴 (BizardNet 方向头) 为冻结清单中唯一带解冻条款的方向 (held-out-pert 诚实验证强制) — 下一步评估其训练可行性 (se600m 状态核查).
- HY 类内部已无 >0.0924 期望点; 盲花第 2 发只会重复 rank 762–791 抖动.

## 2026-10-19 02:12 UTC — iter037-续1: 架构学习轴 (BizardNet 方向头) 可行性判决 — 独立载体全线不过门, 重训设计定为 per-target×per-context 幅度学习 + HY 基底

**证据链** (iter037 记录的「se600m 状态核查」+ 架构资产盘点, 本轮收口):

1. **se600m r2 三臂 (iter035-续 13:35 UTC 已录, 此处引用结案)**: arm_birds 0.2449 / arm_birds_dl 0.1207 / arm_dl -0.8706, 全 FAIL 门1 ≥0.395446; arm_birds 与 round-1 逐位一致 (确定性), 无新信息。se600m-init 的 DL 臂同样崩 (arm_dl -0.8706) — 第二个架构类资产独立提交性证负。
2. **BizardNet skip-head 载体 bench (anchor_v11_sweep.json, official anchor-normalized 口径, Q_HepG2)**: 8 tag λ∈{12,16,18,20} × {v10r,v11} 全读数 — skipv10r_lam12 -0.0241 / skipv11_lam12 -0.0301 / skipv10r_lam16 -0.0105 / skipv11_lam16 -0.0105 / skipv10r_lam18 -0.0070 / skipv11_lam18 -0.0035 / skipv10r_lam20 -0.0093 / **skipv11_lam20 +0.0015 (唯一正值, 仍距 HY8 anchor 0.3154 差 ~0.31)**。anchor pds 全带仅 0.040–0.052 (HY 族 0.31+), nmae −0.40…−0.78, raw mse 1.2–1.9 — 方向头信号存在 (λ↑ 单调改善) 但幅度信噪比差 6 倍, 独立成压不可行。
3. **anchor_v11fix sweep 无读数**: 7 entry 全 error「agg_results.csv 不存在」— 该 sweep 从未跑通 local_bench, 无证据价值。
4. **LOCO 代码判决 (bizardnet.py 审计)**: transformer gated 分量跨上下文稀释信号 (gate≈0, 输出高方差低相关); 20-dim skip 线性头是唯一迁移载体 (λ16 0.3521 vs CCRN 0.3474)。
5. **官方 pds 0.285 vs fid -0.032 分裂**: 方向/形状头达冠军级, 败轴 = 幅度 — 与 iter018 诊断 (per-target×per-context 幅度预测是 redesign 靶心) 一致。

**裁决**:
- **BizardNet 方向头独立提交性证负** (skip 载体距门 ~0.31, se600m 三臂全败) — 冻结清单「BizardNet 方向头」按解冻条款评估完毕: 现有 checkpoint 无可提交产物。
- **重训设计方向 (唯一合法出口, 依 L269 + iter018)**: per-target×per-context **幅度**预测, 直接在 fold/2^Δ 空间学习或以 pds 对齐为目标函数; 输出**必须**与 HY 族构造 (NC 基底 + top-k 收缩让位) 集成作幅度/方向修正头, **永不**独立成压 (skip-head 与 raw-DL 独立提交均已 bench 证负)。
- **解冻条款继续生效**: 任何重训臂必须 held-out-pert 诚实验证 + bench 门 (heur overall_pred > 0.0859 + margin) 过门才花提交。
- 10-19 第 2 发维持不出手 (无候选期望 >0.0924)。

**冻结未动(λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励; BizardNet方向头评估完毕转为重训设计待建)**

## 2026-10-19 06:05 UTC — iter037-续2: L269 幅度头 v1 落地并开训 — per-(target,gene,context) |y| 预测, 双目标函数, hold-100 预注册门 G1–G4

**动作** (依 iter037-续1 重训设计, 唯一合法出口):
- `tmp/bzr1/train_amp_v1.py` 写毕 + py_compile 通过 + 后台启动 (PID 1374102, log `logs/train_amp_v1_1019.log`)。
- 结构: 23 特征 XGB (v2 的 14 列 + 幅度谱 9 列: rk_desc/wd/top8/top50/kth8/kth50/ax_rel_kth8/ntc_cv/ntc_theta), 训练目标 = **|y|** (只学幅度, 方向仍由 HY 构造 sign 门给出 — 冻结条款: 输出只作 HY8 幅度源替换, 永不独立成压)。
- 双目标函数: reg:squarederror (pds/cosine 对齐, 行内大幅度主导) + reg:pseudohubererror (huber_slope 0.05, 稳健对照); hist/cuda, depth 8, eta 0.05, 500 rounds。
- 训练/评估切分: 5 源 (SRC = CTX − hepg2_all) × hepg2 真值交集对; **hold = 100 个 bench 靶 (Q_HepG2 perts, 词汇表码位断言 ==100)**, hepg2 源侧统计同样剔 hold — 无泄漏。
- 基线: |x| (源侧 raw), |yh| (v2 校准器), gene-prior |TGM|; 预注册门 **G1 mae<|x|, G2 sp>|x|, G3 mae<|yh|, G4 sp>|yh|** (G3/G4 = 必须超 v2 校准器的幅度读数)。
- 输出: M_amp_sq.json / M_amp_hu.json / amp_hold_v1.parquet (probe 集成用 hold 明细) / train_meta_amp_v1.json (全部基线+门+verdict)。
- 对齐律执行: np.unique 后无条件 bincount-mean 经 inv 重排 (iter030-修正3); 分母 clamp max(·,1e-3); CALY 缺失源 (jurkat/rpe1/k562_ess 无 _cal.parquet) 置空表不崩。

**通过后路径** (未过门则幅度轴终判): amp_hold_v1 → sim5/sim8c probe 替换 |YH| top-8 括号幅度源 (sign 门不变) → bench 链 Q_HepG2 → heur overall_pred > 0.0859 + margin 才出手 (10-20 额度)。

**冻结未动(λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励; 校准→方向构造族 closed; BizardNet 方向头独立提交 closed→仅作重训材料)**

## 2026-10-19 07:15 UTC — iter037-续4: amp头v1 完成 — G1–G4 双模型全PASS, 幅度轴首次过诚实验证门

**结果** (train_meta_amp_v1.json, hold=100 bench靶 3,740,985 对, 训练 85,114,067 行, t=1739s):
- 修复 2 次 crash (SRC 定义顺序 + 空 CALY searchsorted −1 越界守卫) 后第 3 次启动跑通 (commit 0b97394)。
- **amp-sq**: mae 0.0364 / corr 0.591 / sp_med 0.3130 — G1(beat |x| mae) G2(beat sp) G3(beat |yh| mae) G4(beat sp) **全PASS**。
- **amp-hu**: mae 0.0355 / corr 0.590 / sp_med 0.3098 — G1–G4 **全PASS**。
- 基线: |x| mae 0.0481 sp 0.0763; |yh|(v2校准器) mae 0.0524 sp 0.0374; gene-prior mae 0.0457 sp 0.2020 (模型双指标均超 gene-prior)。
- 注意: gene-prior sp_med 0.202 远超 |x| 0.076 — 幅度谱信息主要来自基因先验+源内排名, 模型在其之上再 +0.11 sp。
- 产出: M_amp_sq.json / M_amp_hu.json / amp_hold_v1.parquet (3,740,985 行 hold 明细 src/target_gene/gene/abs_sq/abs_hu)。

**下一步** (按 iter037-续2 预注册路径): amp_hold_v1 → sim5/sim8c probe 替换 |YH| top-8 括号幅度源 (HY8 sel 门不变, 只换幅度) → probe 门 (d_vs_A ≥ +0.045, dF ≥ −0.001, sanity) → bench 链 Q_HepG2 → heur overall_pred > 0.0859 + margin 才出手 (10-20 00:00 UTC 额度, bizard-v0.11)。

**冻结未动(λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励; 校准→方向构造族 closed; BizardNet 方向头独立提交 closed)**

## 2026-10-19 09:00 UTC — iter037-续5: sim11 集成 probe 判决 — 3 候选过双框架门但效应量噪声级, 进 bench 链验证

**sim11 probe** (`tmp/bzr1/hold_sim11.py`, 表 `tmp/bzr1/sim11_table.csv`, log `logs/sim11.log`):
- 完整性: sim8c 框架 hy8 0.1642 **精确复现** sim8_table (raw 0.1912 / nc 0.1568 同) — sim8c 侧可信; sim5 侧 **S1/S3 FAIL** (hy8 0.8606 > raw 0.8130 方向倒挂; raw 0.8130 vs 历史锚 0.5121±0.10) — sim5 rank-cosine 框架本轮未校准, 只作参考。
- amp_hold_v1 → (p,g) 聚合 → 6146 列映射: coverage agg2=agg5=1.000。
- 15 候选 (KDE=8, sign 门 X*YH>0 不变): ADVANCE 门 (d_vs_A ≥ +0.045 & dF ≥ −0.001 & d_vs_HY8 > 0 & d8 ≥ 0) 恰过 **3 个全 5 源聚合**候选:
  - **amp_max_5src** (括号=|YH|, 幅度=max(|YH|, A_sq_5src)): d8 **+0.0037** (验证框架最大) — **首选**
  - amp_sq_brYH_5src: d8 +0.0032; amp_hu_brYH_5src: d8 +0.0012
  - 全部 2src 变体 d_vs_HY8 > 0 失败 (5src 严格更优 — 模型训练用了 5 源, 聚合一致性非必要)
  - amp_brM_magYH (模型括号) **差于** hy8 (−0.0009/−0.0014) — 括号换模型证负, 选位必须留在 |YH| top-8
- 诚实标注: 效应量 +0.001~+0.0037 为噪声级 (vs 前3差距 3.5×); 与历史 amp8 (+0.0010) 同量级, 4 次迁移失败先例。唯一进 bench 链的理由: sim8c 框架本轮完整性好 + amp_max 是 max-form (只增不减, 风险下界=HY8)。
- TOPK=8 溯源验证 (gwps calHY8 全量 81.2M 行): top-8-|yc| 门在 agree 行内恰 **800 行 = 8/靶 × 100 靶**; HY-state 776 行 per-target 中位恰 8 (并列小差); 2,451 max-form 行 + 24,745 NC-fallback 行 (v==x) — 均为 tie 级差异 (~0.3–0.5%), 重建按 build_construct5 模式 TOPK=8 即可对齐。

**下一步** (预注册): 写 `tmp/bzr1/build_amp_hy8.py` (build_construct5 模式, TOPK=8, 100 bench 靶 sel 位幅度=max(|YH|, A_sq_5src), NaN→|yc|; 其余靶保 |yc|) → `{gwps,hct116_all}_calHY8amp.parquet` → config `bench-bzr1v16-hy8amp-lam70-mf14.yaml` → bench 链 Q_HepG2 → heur overall_pred > 0.0859 + margin 且 >0.0924 官方期望才出手 bizard-v0.11 (10-20 00:00 UTC 额度)。若 bench 无增益 → amp 集成轴 (L269 最后合法出口) 封闭, 重估方向。

**冻结未动(λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励; 校准→方向构造族 closed; BizardNet 方向头独立提交 closed)**

## 2026-10-19 10:05 UTC — iter037-续6: build_amp_hy8 修复+完成 — 双源 HY8 重建 0 差异; 首 bench 因缺 hct116 parquet 作废, 干净重跑中

**build_amp_hy8.py 修复与产物** (脚本 `tmp/bzr1/build_amp_hy8.py`, log `logs/build_amp_hy8.log`, t=1965s):
- IndexError 修复: A-fill 方向翻转 (查 amp_map 端 indexer 而非反), mask 长度=源行数, gwps 一次跑通。
- **gwps**: raw unique 81,162,977; bench靶行=823,011; amp覆盖=739,844 (89.9%); sel&bench=752; **幅度实改=457 (60.8%)**; corr(x,amp)=0.991; **HY8 重建 vs 既有文件: 不一致 0/81,162,977 (0.000%) max|Δ|=0.0000** — 完美复刻, 全链 (load_x/load_cal_y/NC/de_mask TOPK=8/sel/v_hy8) 验证通过。
- **hct116_all**: raw unique 669,741,072; bench靶行=3,356,808; amp覆盖=810,545 (24.2%); sel&bench=684; **幅度实改=320 (46.8%)**; corr=0.996; **HY8 重建 0/669,741,072 不一致 (0.000%)** — 完美。
- 产物: `{gwps,hct116_all}_calHY8amp.parquet` (与 calHY8 同尺寸, 仅 ~457/~320 值不同 — 足迹 = sim11 预测的 top-8×100 靶 sel 位)。

**事故记录**: bench 链 (task byat0pxi6) 在 hct116_all_calHY8amp.parquet 写盘前就启动 (build 仍在 amp-fill 阶段), exit 0 产出 heur overall_pred **0.0296** — 无效读数。bench_bzr1v16hy8amp_Q.log 为空 (0 字节), score 阶段用 bundle 走完; 作废依据: 签名目录 09:45 UTC 无 hct116 parquet, build 05:49 本地才写盘。教训: **依赖后台产物时必须先验证文件存在再启动下游链**。
- 修复后干净重跑已启动 (run2, 两个 parquet 均在场, 515MB+4.2GB mtime 均为 10-19)。

**下一步**: run2 heur overall_pred 判决 — >0.0859+margin 且 >0.0924 官方期望才出手 bizard-v0.11 (10-20 00:00 UTC 额度)。若 ≤ HY8 的 0.0915+margin → amp 集成轴 (L269 最后合法出口) 封闭, 重估方向。

**冻结未动(λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励; 校准→方向构造族 closed; BizardNet 方向头独立提交 closed)**

## 2026-10-19 10:25 UTC — iter037-续7: amp 集成轴判决 — bench heur 0.0876 < HY8 0.0915, **封闭** (L269 最后合法出口关闭)

**bzr1v16hy8amp 干净重跑判决** (run2, 双 parquet 在场, log `logs/bench_v16_hy8amp_run2.log`):
- heur overall_pred **0.0876** (pds 0.5092 / nmae −0.0559 / fid −0.0563 / reach 0.1272 / jac 0.0016; local_anchor 0.3033)。
- vs v14hy8 **0.0915**: Δ = **−0.0039** — sim8c d8 +0.0037 的正读数**未迁移** (第 5 次迁移失败, 与 triple-proxy law 一致)。
- 分量闭环 (family 级 calib a: pds 0.828/nmae 0.280/fid 2.339/reach 0.635/jac 0.639, 分解合计 −0.00390 = 实测): 主掉分 **nmae −0.0030** (local nmae −0.065, from_replicate −0.2568 vs −0.1917) + **pds −0.0013** (−0.0092)。amp 幅度抬高在 Q_HepG2 上让 Wilcoxon lfc 与真值偏离更大 (幅度过高), pds 排序也轻微受损。
- 足迹回顾: 仅 457 (gwps) + 320 (hct116) 个 sel 位改动, 双源 HY8 重建 0.000% 差异 — 构造正确, 是**方向性负效应**, 非 bug。

**判决 (按 iter037-续5 预注册)**: heur 0.0876 ≤ HY8 0.0915+margin, 门未过 → **amp 集成轴封闭**。sim8c 侧 d8 +0.0037 是假阳性 (max-form "只增不减" 对 sim 框架成立, 对真 bench 不成立 — sim 框架对幅度谱的偏好与真评分歧)。
- L269 重训指令的"最后合法出口" (per-target×per-context 幅度学习 + HY 基底集成) 已实测证负。至此 iter037 全部路径 (架构学习轴 sim8 三臂全败 → amp头 G1-G4 PASS 但集成证负) 走完。
- bizard-v0.11 出手取消 (10-20 00:00 UTC 额度不烧, "不为花而花" — 无 >0.0924 期望候选)。

**下一步 (重估方向)**: 1) 官方测试时最大化检查 — 今日 (10-19) 已烧 1 发 (v0.10=HY8 0.092219 rank 764), 今晚 00:00 UTC 新额度窗口决策是否出手 (需新想法过门); 2) 未探索合法方向盘点: HY 状态组合 (HY8 分量最优但 HY3/HY6 邻近, per-metric 权重结构 0.0915→官方 0.0922 映射已最优化?)、五源聚合再变体 (brYH 括号证负已封)、数据侧 (closed)。诚实评估: 若无新合法轴, 保持 0.092378 等待官方榜变化。

**冻结未动(λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励; 校准→方向构造族 closed; BizardNet 方向头独立提交 closed; **amp 集成轴 closed**)**

## 2026-10-19 10:40 UTC — iter037-终: 方向重估完毕 — 全部合法轴确认封闭, hold 0.092378 等官方榜变化

**重估执行** (本 tick, 10:26–10:40 UTC):
1. 官方状态复核: v0.10=HY8 published **0.092219 rank 765** (entry A8L3mpuxVO9GsqtI2JxM, 00:03:06 UTC 提交; rank 从 764 漂至 765, 榜活跃); v0.9=HY12 0.092378 rank 762 仍最佳。今日 (10-19) 已烧 1 发, 第 2 发按"不为花而花" hold。
2. k-curve 原始指标表提取完毕 (summary.json, Q_HepG2): pds 随 k 单调降 (hy3 0.8305→hy50 0.8081), nmae k=12 最优 (0.9454, hy8 0.9456 近平), reach 随 k 单调升 (0.4089→0.4296), fid 平 (0.5815→0.5834), jac 近平。**HY8 heur 峰值 = pds-vs-nmae/reach 的 tradeoff 最优点, 无未消费的 per-metric 冗余**。
3. make_predictions.py 残余旋钮清查: stress_alpha — iter003 既有判决「应激正交化无效」(gwps 全局应激 |mean logFC|≤0.04 低于 DE 分辨率), 无需重测; flip_genes/alignment — 无 HY config 启用, 且属 signature 手术族 (L269 closed); dataset_weights/gene_weights — 均 closed。
4. HY 状态组合假说否决: 混合-k 无机制意义 (k 在构造期一次性施加); per-target 自适应 k 属 per-target 手术 (iter025 量化封死「逐靶收益天花板远低于门」); HY 变体聚合属聚合族 + brYH 括号证负双重封闭。

**终局判决**: 17 条轴全部在档封闭 (λ/幅度/校准→方向构造/签名手术/签名源/k 微调/聚合/置信度链/新数据族/per-target 手术/dataset_weights/方向头独立提交/DL delta/sim10 wrong-drop/brYH/五源聚合/数据侧/amp 集成)。**无 >0.0924 官方期望候选 → bizard-v0.11 不出手, 10-19 第 2 发不出, 10-20 00:00 UTC 窗口按同门重估** (需新证据非旧读数重读, iter025 重置规则)。当前态势 = 保持 0.092378 (rank 762) 等待官方榜变化, loop 转低频守望。

**下一步触发点**: 10-20 00:00 UTC 额度重置 tick 重估; 或任何新证据出现 (官方榜结构变化/新数据族线索)。

**冻结未动(λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励; 校准→方向构造族 closed; BizardNet 方向头独立提交 closed; amp 集成轴 closed; stress_alpha 既有判决不重测)**

## iter037-守望·2 (2026-10-19 18:30 UTC)

**v0.10 rank 漂移加速**: 13:47 UTC rank 770 → 18:28 UTC **rank 780** (4.7h −10 位, 此前速率 ~5位/13h, 加速 ~5×)。六分量分数不变 (pds 0.5082550 / mse 0 / nmae −0.060502 / fid −0.028896 / reach 0.131019 / jac 0.003440), 纯竞争者灌榜。按 iter025 规则非新证据 (无榜结构变化/新数据族), 不出手。含义: 若漂移速率维持, 到 10-20 00:00 UTC 重置窗口 rank 可能至 ~792; 10-20 两发额度价值随漂移升高, 但 gate 不变 — 仅当候选 heur>0.0859+margin 且官方期望>0.0924 才出手, bizard-v0.11 备用名不变。

**下一步触发点**: 10-20 00:00 UTC 额度重置 tick 重估 (需新证据非旧读数重读); 或新证据出现。

**冻结未动(λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励; 校准→方向构造族 closed; BizardNet 方向头独立提交 closed; amp 集成轴 closed; stress_alpha 既有判决不重测)**

## iter037-守望·3 (2026-10-20 00:10 UTC)
**10-20 重置窗口重估**: 10-19 额度重置, 新 UTC 日开始。v0.10 rank 读数链 770→780→779→**778** (23h 净 −8), 漂移速率已衰减至 ~1位/1.5h 以下, 纯竞争者上下榜噪音, 六分量分数全程不变 (Overall 0.09221944692891787 恒定)。无榜结构变化、无新数据族线索 → 按 iter025 规则非新证据, **不出手**。
**判决**: 17 轴封闭判决不变; gate 不变 (heur>0.0859+margin 且官方期望>0.0924); 无 gate 清空候选 → bizard-v0.11 继续备用, 10-20 第 1 发持有 ("不为花而花")。无预置 .vcc 工件 (find 确认)。
**下一步触发点**: 新证据出现 (榜结构变化/新数据族线索/gate 清空候选); 否则继续守望, fallback 1800s。
**冻结未动(λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励; 校准→方向构造族 closed; BizardNet 方向头独立提交 closed; amp 集成轴 closed; stress_alpha 既有判决不重测)**

## iter038 (2026-10-20 07:3x UTC) — NFS 故障恢复 + ECRD 决策探针2 完成: 前提否决, ECRD-no-go

**基础设施恢复**: 2026-10-20 早间 NFS wanglab 子树权限墙被管理员解除 (根因: 用户改了 wanglab 组权限, 但本会话进程组列表为登录时快照、无 gid 1003; 组 ACL 对老进程不生效)。Bash/读写全链路恢复, 无数据损坏。教训已入 memory nfs-repo-outage-20261020。

**ECRD 决策探针2 (tmp/bzarch_probe2.py, 141 eligible targets 中前 40)**:
- 跨 Flex 组 per-target 伪批量 logFC (vs 同组 NTC): spearman 中位 F1-F2 0.385 / F1-F3 0.351 / F2-F3 0.428
- pearson 中位 0.61/0.63/0.66 — 但这是幅度谱共享 (高表达基因普遍 |Δ| 大) 驱动
- **关键否决证据**: 每靶减去三组均值后的 context-specific 残差, 组间 spearman 中位 **-0.48/-0.45/-0.43 (全负!)**
- 解读: 残差完全不可跨组复现 (负相关≈纯噪音甚至反向), "context-specific 可学成分" 为零。probe1 NTC cos 0.9998 的警告被坐实: Flex_1/2/3 本质是同 context 的重复测量, 组间 logFC 差异 = 采样噪音, 不是 context 信号。
- 按探针判据 (spearman~0.9→坏 / 0.5-0.7→好) 实测 0.35-0.43 且残差负 → 落在"最坏"档: 连"跨组均值基线"的增益空间都没有 context 结构可学。

**判决**: ECRD (Embedding-Conditioned Response Decoder, SE-600M embedding 学 Δ(target,context) 迁移) **前提否决, no-go**。vcc2025 Flex 组间无 context-specific 信号 → 学不到"响应如何随 context 变化"。第 18 条轴封闭: **大架构 ECRD / vcc2025 跨 Flex 迁移轴 closed (probe2 残差全负)**。
- 附带含义: vcc2025 与当前挑战仅 25/300 靶重叠 + Flex 组间无 context 结构 → 该数据族对当前 Δ 构造无增量信息 (与 iter029 新数据族 closed 判决一致, 现有机理证据)。
- se600m 三臂先例 (arm_dl -0.894) + 本探针 → "学习式 Δ 替换" 路线在当前数据条件下整体无支撑。

**结论**: bizard-v0.11 继续备用, 10-20 额度 0/2 继续持有 (无 >0.0924 期望候选)。ECRD 探针结论为本轴终审, 除非出现新数据源。
**冻结未动(λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励; 校准→方向构造族 closed; BizardNet 方向头独立提交 closed; amp 集成轴 closed; stress_alpha 既有判决不重测)**

## iter038-babysit (2026-10-20 07:30 UTC) — 心跳核账: 三臂r2终读 + v05在档确认, 无异常

**(1) 三臂bench第二轮已终读** (bzv06dl_r2, lam=2.5): arm_birds 均值 **0.24486** (=round-1 0.24486 复现一致) / arm_birds_dl 0.12066 (round-1 0.1075, 同量级) / arm_dl −0.87058 (round-1 −0.894, 复现)。**门1 (≥0.395446) 三臂全部未过** — DL delta 头纯毒 (arm_dl −0.87), birds+dDL 混合反而拖累 (0.245→0.121), 与 iter037 前判决一致: se600m/birds 学习式 Δ 路线 bench 层面确认无支撑, 与 probe2 残差全负互证。此轴维持 closed, 不再投算力。

**(2) v05 提交状态**: entry 50afg20XylIFiivjWBN9 **published** (0.00066 rank 1028), 非卡 launching — 10-16 日志为历史终态, 无 cancel/重提动作需要。当前最优仍 bizard-v0.9 (HY12) 0.092378 rank 762。

**(3) 额度**: 10-20 已用 0/2, 无 gate 清空候选 → 继续持有。

**(4) 结论**: 心跳三项核账全部闭合, 无待办异常。ECRD/birds/DL 学习式路线三重证据 (bench r1+r2 / probe2 残差负 / 三臂r2复现) 终审封闭。bizard-v0.11 备用, 等新证据。
**冻结未动(λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励; 校准→方向构造族 closed; BizardNet 方向头独立提交 closed; amp 集成轴 closed; stress_alpha 既有判决不重测)**

## iter039 (2026-10-20T10:05Z) mse轴机制判决 — λ微扫闭轴 + 大架构判向

**(1) mse scoring 机制终读** (cell_eval2 scales/scoring): scored mse = clamp(1 − u, 0, 1), u = `expr_mse_unbiased_capped_norm` 官方标度 base=1.0 (control-paste=0分)。23发官方全 0 ⇒ 官方 u ≥ 1.0 恒成立。

**(2) λ 微扫 (HY8 族, Q_HepG2, 5点)**: λ0→u 1.008 / λ0.25→1.0013 / λ0.5→0.9899 / λ1→0.9663 / λ2.5→0.9469 / λ7→0.870 (对照: control-paste 0.765, 复制锚 −0.069)。**全家族 u 下限 0.85**, ×2^(λΔ) 缩放原理上限 ~+0.13 mse 分 (+0.022 overall), 若官方 u 随面板分母收缩 ≥1 则 +0。oracle 臂 (xgbamp_oracle3) u=4.9 更差。⇒ λ-probe 预期收益 +0.000~0.009, **不值得消耗每日发** (额度继续持 v0.11 门)。

**(3) 大架构判向**: leader mse 0.776 ⇔ u=0.224 ⇒ 逐扰动 bulk 均值须比 control-paste 近 2 倍 (L2/18k genes)。pds 0.896 vs 0.508 / nmae 0.413 vs −0.06 / reach 0.457 vs 0.131 — **四轴同源: 全是 Δ 精度**。进入前3的"大架构"不是新头/新损失/新数据集消费, 而是把 Δ̂ 本身从 0.51 拉到 0.85+ 的模型类升级; 现有双源+校准链 18 轴封闭后的残差 = 无跨context可学信号 (probe2 终审)。

**冻结未动(λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励; mse轴λ-probe closed)**

## iter039-续 (2026-10-20T10:22Z) 发1已出: bizard-v0.11 λ2.5 mse轴探针

**(1) 决策**: 用户指令"利用每天的官网提交次数"覆盖 hold 规则 → 10-20 额度发1用于官方 u(λ) 曲线下段采样 (INFO shot)。HY8 族 λ2.5 (本地 Q: u=0.9469/mse_score 0.053/pds 0.774)。

**(2) 执行链**: cfg_lam2.5_submit.yaml → make_predictions (360k cells, counts-preserved, 密度余量 0.56) → vcc prep (targets 300/300 verified) → submit。entry **QerRMb3HOM1R4uZbcnPv**, 提交 2026-10-20T~10:20Z, 上传 3.1GiB/5m16s, scoring job 已起。额度 10-20: 1/2。

**(3) 判读口径** (score 出来后): official mse>0 ⇒ 官方 u(λ2.5)<1 ⇒ λ 维度在官方可兑现 +Δmse; official mse=0 ⇒ 官方面板分母收缩致 u≥1 全域成立, mse 轴对 λ 彻底封闭。同时读 official pds(λ2.5) — 首个官方低 λ 点, 校准 λ-迁移曲线。

**冻结未动(λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励)**

## iter039-终 (2026-10-20T~10:47Z) 发1判读: 官方 mse 轴对 λ 彻底封闭

**bizard-v0.11 (λ2.5) 官方成绩**: Overall −0.0072, rank 1085。分项: **mse=0** / pds 0.2776 (λ70 时 0.5177, 腰斩) / nmae +0.0450 / fid −0.4163 / reach 0.0710 / jac −0.0206。

**(1) mse 判决**: 官方 u(λ2.5) ≥ 1.0 恒成立 ⇒ 官方面板 (真实 400 cells/扰动, 未见面板分母) 使 Δ 缩放全域过 u≥1, 本地 Q 的 u<1 (0.9469) 不可迁移 — 与 official-lambda-shift-law 同族: 幅度类读数本地→官方不可外推。**mse 轴对 λ 彻底封闭, 第 19 条轴**。leader mse 0.776 只能来自 Δ 精度模型类升级 (pds 0.896 同源), 非提交形式工程。

**(2) λ 官方曲线新点**: pds(λ2.5)=0.2776 vs pds(λ47)=0.2956 — 官方 pds 对 λ 在低段平坦 (≈0.28-0.30), 高段 (λ65-70) 跳 0.50+。nmae/fid 低 λ 大幅恶化 (0.045/−0.416 vs λ70 −0.0605/−0.0289) ⇒ 官方钟点在 λ≥65 侧, **低 λ 方向双输, 高 λ 单调占优 — 现行 λ70 已在曲线最优段, λ 链再次确认封闭**。

**(3) 额度**: 10-20 已用 1/2。发2 持有 (无 gate 清空候选; bzr1v16hy8amp heur 0.0876 < 0.0924 期望)。等官方榜 v0.11 rank 漂移观察。

**冻结未动(λ链λ47=bizard-v0.2/affine/gate_report/C2 g/sign_fix关闭/LOCO禁令/PBT奖励; mse轴λ-probe closed; 低λ方向 closed)**

## iter040 (2026-10-20T11:06Z) 官方榜巡检 — v0.11 探针 rank 漂移观察

v0.11 (λ2.5, Overall −0.0072) rank 1085→1088 (55min +3, 与 v0.10 漂移节奏一致 ~1 rank/1.5h, 榜尾涌入)。v0.9/v0.10 entry 已从 status API 不可查 (仅近期 entry 可查), rank 锚用 log 记录: v0.10=764。无新证据, 额度 10-20 1/2 持有, 无 gate 清空候选 (bzr1v16hy8amp 0.0876 < 0.0924)。

## iter040-2 (2026-10-20T11:34Z) 榜巡: v0.11 rank 1088→1089 (28min +1), 漂移正常, 无异常

## iter040-3 (2026-10-20T12:05Z) 榜巡: v0.11 rank 1089→1090 (31min +1), 漂移正常, 无异常

## iter040-4 (2026-10-20T12:36Z) 榜巡: v0.11 rank 1090→1087 (31min −3, 上位清榜/榜尾洗出), 无异常

## iter040-5 (2026-10-20T13:07Z) 榜巡: v0.11 rank 1087→1088 (+1), 漂移正常, 无异常

## iter040-6 (2026-10-20T13:38Z) 榜巡: v0.11 rank 1088→1087 (−1), 漂移正常, 无异常

## iter040-7 (2026-10-20T14:10Z) 榜巡: v0.11 rank 1087 持平, 无异常

## iter041 (2026-10-20T15:05Z) max_fold 首扫 — pds 大幅上涨候选 (mf28), λ90 补充判决

**(1) 背景**: 官方 25 发从未扫过 max_fold (14 永冻)。λ70 时 mf 是次级幅度帽。本地 Q_HepG2 双臂 (HY8 族): mf28 / mf60。

**(2) 结果**: mf14→mf28: **pds 0.699→0.826 (+0.127, anchor-scored 0.557→0.921)**, nmae 0.848→0.948 (恶化), u_mse 0.87→1.74 (官方 mse 已 0 封底 = 免费风险)。mf60≈mf28 (饱和)。λ90 臂因细胞总计数超 1e6 (ScaleLimitError) 先失败 — 已修 make_predictions.py 加 per-cell 总计数帽 (比例收缩至 999,999, 单测通过), λ90 待重跑。

**(3) official_alignment 预测 (mf28)**: overall_pred **0.0918** / pds 0.5173 / nmae −0.0412 / fid −0.0537 / reach 0.1259。vs v0.9 官方 0.092378: 预测持平偏负 (−0.0006), 但校准点均在 mf14 族上 — mf 的官方响应是全新维度 (25 发 0 数据点), local pds +0.127 若官方迁移哪怕 30% → pds 0.52→0.54+。

**(4) 判读**: mf28 是 v0.11 后首个 pds 驱动的候选。gate 需 heur > 0.0859 + margin: 0.0918 过 0.0859 但 < 0.0924 期望。**mf28 vs λ90 组合臂 (λ90+mf28) 为发2 候选** — λ90 单独 +0.002-0.01 预期, 与 mf28 叠加若方向一致 → 有机会破 0.0924。先跑 λ90 修复后 bench。
