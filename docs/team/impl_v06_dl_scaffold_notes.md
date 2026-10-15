# bizard-v0.6 DL Δ̂ 训练脚手架设计笔记（只设计，未开训）

- 日期：2026-10-15；撰写：ML 专家 agent（team-lead 委托）；状态：前置盘点 + 脚手架设计，**未跑任何训练、未占 GPU、未改 src/ 任何代码、未 commit、未 vcc 提交**
- 服务对象：`docs/team/rev_prereg_bizard_v06_dl_delta.md`（R-2 预注册，REV 初审 REVISE 已修订，待平行复审；**过审才开训**）
- 本文只做四件事：盘点 State checkpoints、盘点 Tahoe plate12 就绪度、给出训练脚手架五节设计（数据加载/basal、residual 目标、energy loss、fid BCE 标签、swarm 接入），并披露技术风险。不定义任何训练超参决策权——超参在预注册与训练启动时锁定，本文不越权。

## 0. 盘点一：State checkpoints（可作编码器初始化候选的全部文件）

### 0.1 主家族 ST-HVG-Replogle（官方 State，能量损失 recipe 载体）

根目录 `data/public/models/ST-HVG-Replogle/`（54GB，.done 标记齐全），两臂四上下文：

```
{fewshot, zeroshot}/{hepg2, jurkat, k562, rpe1}/
  config.yaml            # 完整训练 recipe（见 §2 引用）
  data_module.torch      # 序列化 PerturbationDataModule（数据管线与归一化参数）
  pert_onehot_map.pt     # 该上下文扰动→onehot 映射（pert 分支维度来源）
  batch_onehot_map.pkl / cell_type_onehot_map.pkl / var_dims.pkl
  checkpoints/{best,final,last}.ckpt   # 每臂每上下文 3 个 × 450MB，共 24 个 = ~10.8GB
  eval_best.ckpt/ eval_last.ckpt/      # 官方评测输出目录（h5ad/DE csv），非权重
  version_0/hparams.yaml + metrics.csv
  wandb_path.txt
```

- 权重格式：PyTorch Lightning `.ckpt`（单文件 450MB/个）。本地环境齐备：`arc-state==0.11.3`、`torch==2.14.0`、`anndata==0.12.19`。
- **可加载性已被本地实测证明**：`outputs/metrics/local_bench/state_zh_hepg2/`（含 `raw_infer.h5ad`、`Q_HepG2_eval/`、`run_meta.json`、`anchor_normalized.json`）由 `scripts/state_infer_pred.py` + `tmp/run_state_infer.py` 在本机跑通 zeroshot/hepg2 推理（iter019 换算层探针，`docs/iterations/2026-10-12_iter019_state换算层探针.md`、`docs/team/rev_state_st_probe_补充.md`）。fewshot/zeroshot 两臂同构，无理由不可加载；唯一注意点是 ckpt 为 Lightning 容器，加载走 arc-state 的模型构建入口而非裸 `torch.load`。
- 编码器初始化候选池 = 8 个 `checkpoints/best.ckpt`（2 臂 × 4 上下文），择优指标按预注册 §4 在训练启动时锁定一次。

### 0.2 其他候选（存在但不建议作首选）

| 资产 | 路径 | 大小 | 状态 |
|---|---|---|---|
| SE-600M | `data/public/models/SE-600M/`（model.safetensors + se600m_epoch4/16.ckpt + protein_embeddings.pt） | 28GB | 已被 fusion 使用（`emb_se.npy` 5120→PCA-128）；蛋白嵌入可作基因特征，但非 State 训练管线原生 ckpt |
| Tahoe-x1 70M | `data/public/models/Tahoe-x1/70m-model/`（model.safetensors + best-model.pt + vocab.json） | 544MB | 已被 fusion 使用（`emb_tahoe.npy` 62720×512）；作编码器初始化需自建适配层，优先级低于 ST |
| HF 缓存 | `~/.cache/huggingface`：models--arcinstitute--SE-600M / --ST-HVG-Replogle / --tahoebio--Tahoe-x1 | — | 与本地 data/public 副本一致，无需重复下载 |

### 0.3 官方 recipe 关键值（摘自 `zeroshot/hepg2/config.yaml`，四上下文同构）

```yaml
data: PerturbationDataModule
  embed_key: X_hvg            # 模型输入 = HVG 空间（~2000 基因）
  output_space: gene          # 输出映射回全基因空间
  pert_rep: onehot            # 扰动 = onehot（维度来自 pert_onehot_map）
  basal_rep: sample           # ← 样本级 basal（预注册 §2 与此同源）
  n_basal_samples: 1
  basal_mapping_strategy: random   # basal = 随机采样一个对照细胞
  should_yield_control_cells: true
  control_pert: non-targeting
  map_controls: true
  batch_col: gem_group; pert_col: gene; cell_type_key: cell_line
  store_raw_basal: false; int_counts: false; barcode: true
model: state
  cell_set_len: 64; blur: 0.05
  hidden_dim: 328; n_encoder_layers: 1; n_decoder_layers: 1
  loss: energy; distributional_loss: energy   # ← 能量损失
  predict_residual: true                      # ← residual 目标
  softplus: true; batch_encoder: true
  transformer_backbone_key: llama (bidirectional, hidden 328, 8L, 12 heads, intermediate 3072, max_pos 64)
```

## 1. 盘点二：Tahoe plate12 数据就绪度

### 1.1 原始数据（全部在本地，无下载缺口）

| 文件 | 大小 | 内容 |
|---|---|---|
| `data/public/tahoe/plate12.h5ad` | 37.5GB | CSR：**10,487,057 细胞 × 62,710 基因**（`var['gene_name']` 符号，TSPAN6-first），nnz 163 亿（~1,555/细胞） |
| `data/public/tahoe/obs_metadata.parquet` | 2.29GB | 全部 plate 合计 100,648,790 行；plate12 子集 10,487,057 行；列 `plate, BARCODE_SUB_LIB_ID, sample, drug, drugname_drugconc, cell_line(CVCL), sublibrary, pcnt_mito, S_score, G2M_score, phase, pass_filter, ...` |
| `data/public/tahoe/sample_metadata.parquet` | 65KB | sample 级均值质量表（`sample, plate, mean_gene_count, ..., drug, drugname_drugconc`） |

plate12 子集粒度：**95 种药物 × 8+ 细胞系（CVCL 编号）**，剂量并入 `drugname_drugconc`；对照 = `drug` 含 `DMSO` 的细胞。

### 1.2 已加工衍生（重要：签名与嵌入已存在，缺口比预想小）

- **`data/interim/signatures/tahoe_p12.parquet` 已建成**（`src/data/build_tahoe_p12_signatures.py`）：19,918,040 行，列 `[target_gene, gene, logfc]`；`target_gene = "CVCL_xxxx|药名"`（4,700 组 cell_line×drug 伪批量 logFC，vs 同板同系 DMSO 对照，|logfc|>0.05 过滤）。**列名与既有 6 上下文签名 parquet 完全同构**，可直接走既有 fusion 特征管线（当初即为复用而设计）。
- **`data/interim/l15_fusion/emb_tahoe.npy`**（62,720×512，Tahoe-x1 70M gene_encoder 嵌入，PCA-128 入特征）与 `emb_se.npy`（5120→PCA-128）已在 fusion 特征内——DL 头的基因级输入特征可直接复用，不必重提。
- 既有竞赛侧签名 parquet（gwps 516M / hepg2_all 146M / jurkat_all 135M / hct116_all 3.97G 等）与 `feat_train.npz`（10,841,562×19，`y` 即 Δ 目标）——监督信号主表已就绪。

### 1.3 预处理缺口清单

1. **基因空间映射**：plate12 var 为 62,710 个 gene_name 符号；本地六上下文参考面板各不相同（hepg2 bench 9,624 gene_name；hct116 pooled 3.4M 细胞、符号面板；gwps/k562/rpe1/jurkat 各自 h5ad var）。需一次性构建 `plate12 符号 → 各上下文面板` 的对齐表并审计覆盖率（预计高度重叠但须量化，含同义名/版本后缀冲突）。已有 `build_tahoe_p12_signatures.py` 的读取骨架可复用（chunk 200k 流式求和模式，37.5GB 可内存安全遍历）。
2. **basal 对照池**：plate12 DMSO 组即 basal 池（每 cell_line×plate 一组）；需按 §3.1 的 sample 语义重组（同板同系 DMSO 随机采样）。
3. **粒度确认**：预注册说"Tahoe plate12 作辅助训练数据"——现成粒度是伪批量签名（组级），不是细胞级。DL 头若按官方 energy loss 训练需要细胞分布；若按辅助成员 Δ̂ 直出训练则签名粒度够用。**推荐主路线：细胞级输入 + 伪批量 residual 目标混合**（见 §3.4），签名路径零预处理。
4. **词表语义**：`target_gene` 承载的是药名非靶基因——辅助任务与竞赛 300 靶不同语义空间， fid BCE 标签构造（§4）必须只用竞赛上下文数据，Tahoe 仅入主损失支路。

## 2. 脚手架设计（五节）

### 2.1 数据加载布局与样本级 basal 表征构建

**目录布局（新增，均在 data/interim/ 与 src/ 之外动工——src 不改，新训练代码放独立模块，建议 `src/dl_v06/` 新目录，本文只设计不落码）：**

```
data/interim/dl_v06/
  basal_cache/{ctx}.npz          # 每上下文对照池：X_ntc (n_ntc × G) + cell index
  pairs/{ctx}.npz                # (target, basal_idx, Δ_true) 训练对
  gene_align/{ctx}_plate12.json  # 基因空间对齐表
  splits.json                    # 固定验证集（预注册 §4：训练启动时锁定一次）
```

**样本级 basal 构建（对照官方 recipe 逐项）**：

- 官方：`basal_rep: sample` + `basal_mapping_strategy: random` + `n_basal_samples: 1` = 每个 (扰动, 细胞) 对的 basal 是**从该上下文对照池随机抽 1 个对照细胞**的表达（不是池均值）。训练时每 epoch 重抽 → 隐式数据增广。
- 本地复刻：每上下文从 bench 参照 h5ad 的 NTC 池（`local_bench.prepare_reference` 产物已存在）取对照细胞矩阵作 `basal_cache`。训练对 (target, gene-panel Δ) 构建时，basal 采样沿用 random 策略；**推理期** basal = NTC 池均值行（与现行 `ntc_mean` 口径一致，`src/eval/local_bench.py:87-91`），保证与 v0.5 基线同评测口径。
- 作用域（预注册 §2 红线）：basal embedding 只进 DL 成员的条件化输入；既有 6 鸟的 19 列特征（`feat_train.npz`）与 linear-skip 载体输入一字不动。

### 2.2 residual 目标计算

- 目标语义 = 官方 Δ̂ 同构：`Δ(gene) = log2FC(扰动 vs basal)`，伪批量签名已有（竞赛 6 上下文 = `data/interim/signatures/*.parquet`；Tahoe aux = `tahoe_p12.parquet`）。
- 模型输出 `Δ̂ = f(basal_emb, target_emb) − basal_emb`（`predict_residual: true` 的语义：网络预测对 basal 的偏移，不是绝对表达）。
- 与融合链的对接点：DL 成员的 Δ̂ 直接走既有 affine 前的 Δ̂ 通道（与 6 鸟同网格：target×gene，行对齐 `feat_train.gpos` 网格）；**不做任何 λ/clip 事后处理**（幅度头禁令 + λ 链冻结，rank 平均见 §4）。
- 缺失处理：签名 parquet 中 |logfc|≤0.05 行被建表时过滤——训练对构建时以面板全基因重算（从 h5ad 直接出 logFC），避免过滤偏置进入监督信号；仅 fid BCE 标签可用过滤版。

### 2.3 energy loss 实现要点（官方 recipe 原样采用，引用位置）

**引用位置（本文核验过）：**
- 官方配置：`data/public/models/ST-HVG-Replogle/zeroshot/hepg2/config.yaml` — `model.kwargs.loss: energy`、`distributional_loss: energy`、`blur: 0.05`、`cell_set_len: 64`、`softplus: true`、`predict_residual: true`；data 侧 `basal_rep: sample`、`basal_mapping_strategy: random`、`n_basal_samples: 1`、`embed_key: X_hvg`、`output_space: gene`。
- 数据模块与损失实现：随 ckpt 附带的 `data_module.torch` + `pert_onehot_map.pt`/`var_dims.pkl` 可直接被 arc-state 0.11.3 反序列化（iter019 已实测推理路径）。
- 本地旁证：`docs/iterations/2026-10-12_iter019_state换算层探针.md`、`docs/team/rev_state_st_probe_补充.md`（换算层坑：库大小归一/rint 离散化）。

**实现要点（设计原则：不重造能量损失）**：
1. 训练循环用 arc-state 的 State 模型类 + 其 energy loss 原样实例化（`loss: energy`），禁止自写 loss 变体——预注册 §2 明文"官方 recipe 原样采用，不自行改造"。
2. energy loss 需要**扰动组的细胞集合分布**（cell-set 预测 vs 对照分布的能量距离），故数据加载必须保留细胞级 mini-batch（`cell_set_len: 64` 官方值），伪批量签名只作 residual 监督正则支路（辅助 MSE/该支路的具体权重属训练期自由度，按预注册 §4 不入预注册）。
3. `init_from` 字段在 config 中为 null——编码器初始化候选即通过该入口注入（预注册 §4：候选间以锁定验证集单一指标择优）。
4. 输出空间：`output_space: gene` 映射回全基因面板后与 Δ̂ 通道拼接；`softplus: true` 保证输出为非负增益语义（与 `2^(λ·Δ̂)` 载体兼容的下游不变）。

### 2.4 fid BCE 辅助头：DE 方向标签构造

- **标签来源（只用竞赛上下文数据）**：每上下文现成 `*_real_de.csv`（如 `data/public/models/ST-HVG-Replogle/zeroshot/hepg2/eval_best.ckpt/hepg2_real_de.csv`）与 `data/interim/signatures/{ctx}.parquet` 的 `logfc` 列。
- **构造规则（草案，训练启动时随验证集一并冻结）**：对每个 (target, gene)：`y = 1[logfc < −τ]`（下调=1）与 `y = 1[logfc > +τ]` 二选一（建议合成单头：正类=显著下调，其余=0；τ 取现有 DE 管线的显著性阈，与 fid 的方向保真定义对齐）。非 DE 基因不进 BCE（避免过半噪声标签稀释方向信号）。
- **与官方 fid 的耦合**：官方 fid 斜率 2.797（affine v5）是七指标中最大放大器，BCE 头输出 `P(方向)` 作为 DL 成员 Δ̂ 的符号置信重加权（softmax 或直接乘性调制，具体形式训练前随消融表冻结），主梯度仍来自 energy loss——辅助头只是方向正则，不替代主损失。
- **禁止项自查**：BCE 不使用幅度信息（只吃 sign/显著性标签）→ 不构成幅度头回归；不触碰已发布链输出 → sign_fix 禁令无涉。

### 2.5 DL 头接入 swarm.py soft_rank_average（接口已只读确认，不改代码）

`src/models/bizard/swarm.py:98-134` `soft_rank_average(preds: list[np.ndarray], weights=None, reference=None)`：

```python
P = np.stack([p.astype(np.float64) for p in preds])   # (B, N) 每鸟一行
...每鸟→经验分位 q（平均秩处理并列）...
qm = (w[:, None] * q).sum(axis=0)                     # 加权平均分位
ref = P.reshape(-1)（或外部 reference）排序池
return ref 分位插值回 Δ̂ 标度, float32
```

**接入要点（全部由接口性质决定，零改动既有成员）**：
1. **行对齐硬约束**：每鸟 pred 必须同一 target×gene 网格（docstring 明示"行对齐"）。DL 头推理输出须 reindex 到与 6 鸟完全相同的 (target, gene) 顺序——对齐表在 §2.1 的 `gene_align/` 预生成并逐位断言。
2. **调用侧改动极小**：聚合入口 `aggregate(preds, method="soft_rank", weights)`（swarm.py:146-152）只看 `preds` 列表长度——v0.6 交付物 = 第 7 个 np.ndarray（DL Δ̂）append 进 preds 列表；swarm.py 本体、6 鸟训练、linear-skip 载体一行不动。建议落点为新脚本（如 `scripts/dl_v06_swarm_member.py`，本文不落码）读 `agg_delta_{ctx}.parquet` 现有 6 鸟产物 + DL 推理 parquet，就地 stack 调 `soft_rank_average`。
3. **rank 语义红利**：soft_rank_average 只消费每鸟 Δ̂ 的**秩**（经验分位），不消费绝对幅度——DL 成员无需任何幅度标定即可入群，天然满足"无幅度头"预注册条款；参照分布自动扩为 7 鸟合并池（reference=None 路径），尺度由池内既有 6 鸟主导，DL 只贡献排序信息。
4. **消融表出口**：门槛 4 的三臂（DL-only / XGB+DL / v0.4）= preds 传 `[dl]` / `[6鸟+dl]` / `[6鸟]` 三种列表分别 aggregate——同一接口三臂齐备，无需旁路代码。

## 3. 最大技术风险（单条，按任务要求）

**基因空间与扰动空间的双重错配**：ST checkpoints 的输入空间是 `embed_key: X_hvg`（每上下文自己的 ~2000 HVG，`var_dims.pkl` 锁维度）、扰动分支是各自上下文的 `pert_onehot_map.pt`（本地上下文训练扰动的 onehot 表）；而竞赛面板与 300 靶都不在其中，gwps/hct116 两上下文甚至没有官方 checkpoint。这意味着"编码器初始化"只能迁移 encoder 主干权重，pert 嵌入与输出头必须跨空间重训（HVG→各上下文面板逐一对齐 + onehot 表重建），初始化收益可能远小于表面预期，且存在静默错位风险（同名基因不同 HVG 集合）——建议开训前先做一个只读对齐审计脚本量化 8 个候选 ckpt 与 6 上下文面板的基因交集率，作为编码器初始化候选的硬证据（该审计属不开训准备工作，可先做）。

## 4. 合规自查

- 未跑训练、未占 GPU（本轮全部为 ls/pyarrow/h5py 只读盘点）；
- 未改 src/ 任何代码（本文为纯设计文档；swarm.py 仅只读）；
- 未 git commit、未 vcc 提交；
- 文档唯一写入 = 本文件（team-lead 指定交付路径）。
