"""v0.6 DL training launcher — arc-state 0.11.3 `state tx train`.

Official ST recipe: gwps(K562)+rpe1 joint, hidden 328 / cell_set_len 64 / 8L llama
backbone, lr 1e-3, batch 64, energy loss, batch_encoder true, residual target.
Encoder-init: init_from official fewshot/k562 best.ckpt -> filtered load_state_dict
(strict=False) with decoder + pert_encoder auto-rebuild on shape mismatch
(PI ruling B Option a semantics, implemented inside state/_cli/_tx/_train.py).

LlamaConfig.validate bypass kept from tmp/run_state_infer.py: torch.load of the
legacy official ckpt unpickles a LlamaConfig written by an older transformers;
the no-op validate removes a whole failure class at zero cost.
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
    "model.kwargs.init_from=/data/resource/wanglab/aivc/aivc2026/data/public/models/ST-HVG-Replogle/fewshot/k562/checkpoints/best.ckpt",
    "training.lr=0.001",
    "training.batch_size=64",
    "training.max_steps=100000",
    "training.val_freq=2000",
    "training.ckpt_every_n_steps=4000",
    "training.train_seed=42",
    "training.use_mfu=false",
    "name=state_v06_gwps_rpe1",
    "output_dir=/data/resource/wanglab/aivc/aivc2026/data/interim/dl_v06/run_v06",
    "use_wandb=false",
    "overwrite=true",
]

from state.__main__ import main

main()
