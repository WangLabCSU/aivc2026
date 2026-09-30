"""Wrapper: bypass huggingface_hub strict LlamaConfig validation for arc-state 0.11.1 legacy ckpt."""
import sys
from huggingface_hub import dataclasses as hf_dc
from transformers import LlamaConfig
LlamaConfig.validate = classmethod(lambda cls, self=None: None)
sys.argv = ["state", "tx", "infer",
    "--model-dir", "data/public/models/ST-HVG-Replogle/zeroshot/hepg2",
    "--checkpoint", "data/public/models/ST-HVG-Replogle/zeroshot/hepg2/checkpoints/best.ckpt",
    "--adata", "tmp/state_zh_input_hepg2.h5ad",
    "--pert-col", "target_gene",
    "--control-pert", "non-targeting",
    "--embed-key", "X_hvg",
    "--output", "outputs/metrics/local_bench/state_zh_hepg2/raw_infer.h5ad"]
from state.__main__ import main
main()
