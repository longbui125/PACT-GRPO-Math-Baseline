"""One shared MATH-Algebra warm start for the four math_sft RL runs."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import format_math_prompt, format_prompt
from .modeling import load_tokenizer, load_trainable_model, trainable_parameters
from .radical_equations import RadicalCase
from .utils import append_jsonl, set_seed, write_json


def train_math_warm_start(config: ExperimentConfig, output_dir: str | Path,
                          data_path: str | Path, *, epochs: int = 1) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    examples = [
        json.loads(line) for line in Path(data_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not examples:
        raise ValueError("MATH-Algebra SFT file is empty; run prepare_math_data.py")
    set_seed(config.seed)
    tokenizer = load_tokenizer(config.model_name)
    model = load_trainable_model(config)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=2e-5)
    metrics_file = output / "train_metrics.jsonl"
    metrics_file.unlink(missing_ok=True)
    started = time.time()
    rng = random.Random(config.seed)
    order = list(range(len(examples)))
    model.train()

    for epoch in range(epochs):
        rng.shuffle(order)
        for position, item_index in enumerate(order, 1):
            item = examples[item_index]
            prompt = format_math_prompt(item["problem"], tokenizer)
            response = (
                f"{item['answer']}}}\nExplanation: {item['solution']}"
                + tokenizer.eos_token
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            response_ids = tokenizer(response, add_special_tokens=False).input_ids
            if len(prompt_ids) + len(response_ids) > 256:
                raise ValueError(f"Overlong MATH example {item['source_index']}")
            ids = torch.tensor([prompt_ids + response_ids], device="cuda")
            labels = torch.tensor([[-100] * len(prompt_ids) + response_ids], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=labels, use_cache=False).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            optimizer.step()
            append_jsonl(metrics_file, {
                "epoch": epoch + 1, "sample": position, "source_index": item["source_index"],
                "loss": float(loss), "elapsed_seconds": time.time() - started,
            })
            if position % 25 == 0:
                print(f"[math_sft] {position}/{len(order)} loss={float(loss):.3f}", flush=True)

    adapter = output / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    write_json(output / "summary.json", {
        "method": "math_sft", "source": "HuggingFaceH4/MATH algebra train",
        "samples": len(examples), "epochs": epochs, "elapsed_seconds": time.time() - started,
    })


def train_candidate_warm_start(config: ExperimentConfig, output_dir: str | Path,
                               cases: list[RadicalCase], *, epochs: int = 2) -> None:
    """Teach the old verification family before testing RL transfer."""
    if not cases or epochs < 1:
        raise ValueError("SFT needs cases and at least one epoch")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    tokenizer = load_tokenizer(config.model_name)
    model = load_trainable_model(config)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=2e-5)
    rng = random.Random(config.seed)
    order = list(range(len(cases)))
    metrics_file = output / "train_metrics.jsonl"
    metrics_file.unlink(missing_ok=True)
    started = time.time()
    model.train()
    for epoch in range(epochs):
        rng.shuffle(order)
        for position, index in enumerate(order, 1):
            case = cases[index]
            prompt = format_prompt(case.example.question, tokenizer)
            response = f"{case.example.answer}}}" + tokenizer.eos_token
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            response_ids = tokenizer(response, add_special_tokens=False).input_ids
            if len(prompt_ids) > config.max_prompt_tokens:
                raise ValueError(f"Overlong transfer prompt: {case.example.uid}")
            ids = torch.tensor([prompt_ids + response_ids], device="cuda")
            labels = torch.tensor([[-100] * len(prompt_ids) + response_ids], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=labels, use_cache=False).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            optimizer.step()
            append_jsonl(metrics_file, {"epoch": epoch + 1, "sample": position,
                                        "case": case.example.uid, "loss": float(loss),
                                        "elapsed_seconds": time.time() - started})
            if position % 50 == 0:
                print(f"[transfer_sft] {epoch + 1}/{epochs} "
                      f"{position}/{len(cases)} loss={float(loss):.3f}", flush=True)
    adapter = output / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    write_json(output / "summary.json", {"method": "transfer_sft", "source": "easy_train",
                                            "samples": len(cases), "epochs": epochs,
                                            "elapsed_seconds": time.time() - started})
