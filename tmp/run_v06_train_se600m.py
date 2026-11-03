"""v0.6 DL training launcher — se600m-init ablation arm (metric_lock follow-up).
Clone of tmp/run_v06_train.py with ONLY init_from/name/output_dir changed —
same data config (train_config.toml), same hyperparams (lr 1e-3, batch 64,
max_steps 100000, seed 42). This is the metric_lockTier-A winner arm
(se600m_retrain sweep -0.1698, rank 1/9) vs the live fewshot/k562 arm
(rank 6/9) = gate4 消融第二臂. Per metric_lock disposition:
"se600m_retrain-init arm designated follow-up once GPU frees; NO silent swap mid-run".
Launched ~17:55Z by team-lead directly (ownership rule), coexisting with the
CPU/GPU-light three-arm bench (cell-eval2 ~5.8GB + training ~6GB << L20 46GB).
"""
import sys
from huggingface_hub import dataclasses as hf_dc
from transformers import LlamaConfig
LlamaConfig.validate = classmethod(lambda cls, self=None: None)
sys.argv = [
    "state", "tx", "train",
    "data.kwargs.toml_config_path=/data/resource/wanglab/aivc/aivc2026/data/interim/dl_v06/train_config.toml",
    "data.kwargs.batch_col=batch",
    "data.kwargs.cell_type_key=cell_line",
    "data.kwargs.control_pert=non-targeting",
    "data.kwargs.embed_key=X_hvg",
    "data.kwargs.output_space=gene",
    "model.kwargs.cell_set_len=64",
    "model.kwargs.hidden_dim=328",
    "model.kwargs.batch_encoder=true",
    "model.kwargs.init_from=/data/resource/wanglab/aivc/aivc2026/data/interim/state_retrain/run_v2/state_se600m_gwps_rpe1/checkpoints/final.ckpt",
    "training.lr=0.001",
    "training.batch_size=64",
    "training.max_steps=100000",
    "training.val_freq=2000",
    "training.ckpt_every_n_steps=4000",
    "training.train_seed=42",
    "training.use_mfu=false",
    "name=state_v06_gwps_rpe1_se600m",
    "output_dir=/data/resource/wanglab/aivc/aivc2026/data/interim/dl_v06/run_v06_se600m",
    "use_wandb=false",
    "overwrite=true",
]
from state.__main__ import main
main()
