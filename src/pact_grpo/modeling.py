from __future__ import annotations

from contextlib import nullcontext
from typing import Iterator

import torch
from peft import LoraConfig, PeftModel, get_peft_model
import transformers.utils.import_utils as _transformers_import_utils

# This project is text-only. Disable an optional audio import that is broken
# in the existing tf_gpu environment, without changing installed packages.
_transformers_import_utils._librosa_available = False

from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available() and name in {"bfloat16", "float16"}:
        return torch.float16
    return torch.float32


def load_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_trainable_model(config: ExperimentConfig):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for training.")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=resolve_dtype(config.dtype), trust_remote_code=True, low_cpu_mem_usage=True
    ).cuda()
    if config.gradient_checkpointing:
        # Reentrant checkpointing detaches the frozen base-model input in
        # this PEFT setup, silently yielding no LoRA gradients. The
        # non-reentrant variant preserves the autograd graph.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=config.lora_targets, bias="none", task_type="CAUSAL_LM"
    )
    return get_peft_model(model, lora)


def load_evaluation_model(config: ExperimentConfig, adapter: str | None):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=resolve_dtype(config.dtype), trust_remote_code=True, low_cpu_mem_usage=True
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters found")
    return parameters


def parameter_slices(parameters: list[torch.nn.Parameter]) -> Iterator[tuple[torch.nn.Parameter, slice]]:
    offset = 0
    for parameter in parameters:
        end = offset + parameter.numel()
        yield parameter, slice(offset, end)
        offset = end
