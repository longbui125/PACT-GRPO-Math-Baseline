"""Load the common SFT LoRA checkpoint for radical-equation RL runs."""

from __future__ import annotations

import torch
from peft import PeftModel

from .config import ExperimentConfig
from .modeling import resolve_dtype
from transformers import AutoModelForCausalLM


def load_trainable_adapter_model(config: ExperimentConfig, adapter: str):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for training.")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=resolve_dtype(config.dtype),
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).cuda()
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False
        model.enable_input_require_grads()
    return PeftModel.from_pretrained(model, adapter, is_trainable=True)
