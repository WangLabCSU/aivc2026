# VCC 2026 终局存档 (2026-11-03 定稿)

> 用途: 竞赛收官后的完整可复用档案 — 决策链、提交台账指针、封闭轴清单、复现路径、运维手册。
> 下届/后续项目直接从本文档 + `submission_ledger.csv` + `docs/team/impl_v06_progress.md` 恢复全部上下文。

## 1. 竞赛时间线 (2026 季)

| 日期 (UTC) | 事件 |
|---|---|
| 09-28 | 首发探索开始 (L0 锚点), 验证面板 `vcc2026-val-1` (context A/B/C) |
| 10-05 | bizard (BizardNet skip 载体) 家族首秀 |
| 10-18 | **bizard-v0.9 (HY12×λ70) 官方 0.092378 rank 762 — 系列最高** (验证期) |
| 10-22 | **最终测试集发布**: `controls-final`, context D/E/F, 360k cells × 18533 genes, 面板 `vcc2026-test-1`, 锚 testD/E/F-r1 |
| 10-25 14:20 | final 期第 7 发 v0.21 published, 进入静默保持 |
| 11-03 03:22 | v0.22 演练复刻 (CLI 0.3.0 链路保温 + last 刷新) |
| **11-04 02:39** | **闭合弹 v0.23 提前发射 (用户指令, watchdog 链), 03:09 published — entry beDqtDiHkdbzT5fCcYBy** |
| **11-05 23:59** | **截止**; 截止前受理、在途 processing 仍计奖 |
| 11-19/20 | 虚拟授奖仪式, final 成绩解卷 |

规则要点: 每日 2 发 (UTC 日重置), **只有最后一发计奖**; final 成绩全程隐藏 (盲射)。

## 2. 最终提交决策链

- **弹药**: `outputs/submissions/20261024_bizard_v18_final_hy8_nogeneric.prep.vcc` (3.19GB, 10-24 build)
  - HY8×λ70 nogeneric; heur overall_pred **0.0915 = 全网格最优** (hykens 0.0914 / HY6 0.0907 / HY12λ62、HY12λ78 更低)
  - v0.18/0.19/0.20 网格 5/6 点已官方盲发覆盖; v0.21–v0.23 为同 bundle 身份冗余复刻
- **为何复刻**: 只最后一发计奖 → 闭合弹必须落在"网格最优 × 链路已验证"的组合上; 无反馈信号 (盲射), 网格已闭, 无更优候选
- **为何 at + watchdog**: 裸 sleep-loop timer 与后台链均会死于 session 退出 (11-02 两例实证); `at -t 202611042002` (本地 EST) = UTC 11-05 01:02, atd 存活且重启补跑; `scripts/submit_final_watchdog.sh` 内置 2h phase 停滞 cancel+refire / flock 幂等 / GUARD 护栏

## 3. bizard 系列官方成绩全表 (验证期 + final 期)

| model | 日期 | entry | 配置 | Overall | Rank | 备注 |
|---|---|---|---|---|---|---|
| bizard-v0.2 | 10-14 | BWy91Q4UXdusW6QcxjCU | v02族 λ47 | 0.04565 | 908 | 首个 plain 命名 |
| bizard-v0.3 | 10-15 | SgXJeIQwv7taWP4B0GND | λ37 | 0.05021 | 886 | v02 族峰 |
| bizard-v0.4 | 10-15 | 58kzxoM9gGrF1QOdXS5k | λ35 | 0.04912 | 887 | |
| bizard-v0.5 | 10-16 | 50afg20XylIFiivjWBN9 | **坏 bundle** (stale v04_agg bit-identical) | 0.00066 | 1028 | 事故: 重传前必须核对 bundle |
| bizard-v0.6 | 10-16 | lrHxrp4sceF58JqMstK3 | dualsrcdeep λ75 | 0.08094 | 789 | pds/fid/reach/jac 四纪录; 官方 λ 峰确证 6.5 |
| bizard-v0.7 | 10-17 | 2a8UWn0MwC1l3Xg7ReSJ | dualsrcdeep λ70-mf14 | 0.085809 | 782 | |
| bizard-v0.8 | 10-18 | 9XItMxvCobntWFW6h5zt | hcx 换源探针 | 0.085867 | 788 | |
| **bizard-v0.9** | **10-18** | **CV9sGkgWRLsWCPlPmqve** | **HY12×λ70** | **0.092378** | **762** | **验证期系列最高** |
| bizard-v0.10 | 10-19 | A8L3mpuxVO9GsqtI2JxM | HY8×λ70 | 0.092219 | 764 | |
| bizard-v0.11 | 10-20 | QerRMb3HOM1R4uZbcnPv | λ2.5 (mse 轴探针) | −0.0072 | 1085 | mse 轴封闭 |
| bizard-v0.12 | 10-20 | 8GoyezvGtSKDD7J8pubb | HY8×λ70+mf28 | 0.092247 | 783 | mf 轴封闭 |
| bizard-v0.13 | 10-21 | u2oI7b7p8tY9cWhQWyKU | HY12+vcc2025 注入 | 0.03333 | 984 | 跨 context 注入证伪 |
| bizard-v0.14 | 10-22 | AU6y2WVVRPxOfx6AD7xp | final v09replica (generic-on) | 盲 | — | final 流程打通 |
| bizard-v0.15 | 10-22 | 8RozCc72GWn0w93y4qqu | final nogeneric | 盲 | — | 与 v14 构成盲发对照 |
| bizard-v0.16 | 10-23 | YHqgjInyD7Aoe1H2lVku | HY6×λ70 | 盲 | — | 网格点 |
| bizard-v0.17 | 10-23 | 8PWIw0tv4ATZnfNTkWD8 | HY12×λ62 | 盲 | — | 网格点 |
| **bizard-v0.18** | **10-24** | FKkQQh21MIHwTrhE8bHx | **HY8×λ70 nogeneric** | 盲 | — | **网格最优 bundle, 终局弹药源** |
| bizard-v0.19 | 10-24 | Yt94J8oI8BShjjOuraHH | HY12×λ78 | 盲 | — | 网格点 |
| bizard-v0.20 | 10-25 | OfOYrkK5HQa0nENXyjlY | hykens | 盲 | — | 网格点 (0.0914) |
| bizard-v0.21 | 10-25 | dPBxnKlPGsLATyp75WSi | =v18 复刻 | 盲 | — | last 升至网格最优 |
| bizard-v0.22 | 11-03 | 7hpdoYjVOSgydX5KBDUB | =v18 复刻 | 盲 | — | 演练 + CLI 0.3.0 保温 |
| **bizard-v0.23** | **11-04** | **beDqtDiHkdbzT5fCcYBy** | =v18 复刻 | 盲 | — | **闭合弹 published 03:09 UTC = 最后一发计奖发** |

更早期启发式/fusion/ens 时代 (09-28 ~ 10-09, l0→ens_alpha05) 见 `submission_ledger.csv` 前段; 官方快照全集 `outputs/metrics/*_online.json`。

## 4. 封闭轴清单 (22+ 轴, 全部官方实测或本地闭环)

λ 网格 (35–78) / max_fold 帽 / mse 项 / 签名源九族枚举 / k 微调 / 聚合方式 / 置信度链 / 新数据族 / per-target 手术 / dataset_weights / 方向头独立提交 / DL delta / brYH / 五源聚合 / amp 集成 / vcc2025 跨 context 注入 / ECRD 跨 Flex 残差 / 官方 300 靶覆盖 (仅 gwps272+hct116_all282 有交集)。

核心定律 (memory 存档):
1. **official-lambda-shift-law**: 幅度类超参官方迁移只能同族同架构实测; ×1.4 跨族外推律已证伪; 官方读数以 machine snapshot 为准
2. **official300-coverage-law**: bench 通过 ≠ 提交有效 (bench targets 2394 vs 官方 300 几乎不相交); 加源先查覆盖+信息非冗余
3. **official-mse-axis-closed / official-mf-axis-closed / official-vcc2025-inject-axis-closed / ecrd-crossflex-nogo**: 各轴终审记录

## 5. 复现路径 (final bundle)

```bash
# 1) 数据: vcc datasets download controls-final → data/raw/controls_final/
#    (300 靶与验证期 controls 完全换血, 勿混用; gene_names.csv 18533 基因)
# 2) 生成: config configs/l1_signature/bzr1v20-final-hy8-nogeneric.yaml (controls_dir: data/raw/controls_final)
#    → make_predictions.py 链 → h5ad 360000×18533 raw counts, context D/E/F 各 120k
# 3) 打包: vcc prep <h5ad> -g data/raw/controls_final/gene_names.csv \
#         --perts data/raw/controls_final/pert_counts.csv -o <prep.vcc>
#    (CLI ≥0.3.0; targets 自动对官方清单校验; counts-preserved)
# 4) 提交: bash scripts/submit_final_watchdog.sh <prep.vcc> bizard-vN.N <log.abs> "<GUARD UTC>"
#    (内置: keyring 防挂 / 全调用 timeout / 2h 停滞 cancel+refire / flock 幂等 / 护栏后只等终态)
```

环境: conda env `aivc2026`, vcc CLI `/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc` (0.3.0)。

## 6. 工件与目录索引

- `outputs/submissions/*.prep.vcc`: v14–v22 弹药全档 (3.0–3.2GB × 7+); **v18 = 终局弹药**
- `outputs/metrics/*_online.json`: 官方分数 machine snapshot 全集 (权威读数源)
- `logs/submit_bizard_v*.log`: 每发完整链路日志 (含 entry/published 时间线)
- ~~`tmp/QUARantine_dup/`: 7.1G 重复 bundle 隔离区~~ (**2026-11-10 已删**: bit-identical 双胞胎,
  正主 `outputs/submissions/20261016_dualsrcdeep_lam70mf14.vcc` sha256 `019b95422af18ab6` 删前已验证在盘;
  v0.5 事故本身以台账 + docs 记录为准, 大文件不再占盘)
- `tmp/bzr1/` (~130M, **保留**): bzr1 校准器训练链全部脚本/日志/中间件 — iter043-140 已入库,
  其中 >15MB 权重矩阵 (M_ctx*.json / M_amp_*.json) 与缓存 (sim5_cache.npz 等) 按 gitignore 留 NFS 不进 git
- `tmp/state_zh_input_hepg2.h5ad` (33M, **保留**): Q_HepG2 bench 输入快照 (gitignore, NFS 留盘)
- `docs/team/impl_v06_progress.md`: iter 全量过程日志 (append-only 台账)
- `submission_ledger.csv`: 提交台账 (核心协作文件, 本次已补全至 v0.23)
- 关键脚本: `scripts/submit_final_watchdog.sh` (终局发射链), `tmp/launch_timer_*.sh` (历史定时器, 已废弃勿再用)

## 7. 解卷日 (11-19) 阅读指南

- final 期 7 发 (v0.14–v0.22) + 闭合弹 v0.23 全部为盲分; 解卷后对照 §3 本地 heur overall_pred (HY8 0.0915 / hykens 0.0914 / HY6 0.0907)  → 校准 local↔final-panel 迁移矩阵
- v0.14 (generic-on) vs v0.15 (nogeneric) 单变量对照 → generic guides 处理的官方增益读数
- HY 网格 6 点 → HY 构造族的官方响应面; 与验证期 (HY12 0.092378 > HY8 0.092219) 对比 → 检验 context D/E/F 是否翻族
- rank 无排行榜 (盲射), 唯一排序信号 = 官方公布的名次
