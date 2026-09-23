from __future__ import annotations

import time
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import MathExample, extract_final_answer, format_prompt, load_gsm8k, load_math500
from .modeling import load_evaluation_model, load_tokenizer
from .utils import append_jsonl, write_json


@torch.no_grad()
def evaluate_examples(model, tokenizer, examples: list[MathExample], config: ExperimentConfig, output: Path) -> dict:
    output.unlink(missing_ok=True)
    correct, started = 0, time.time()
    device = next(model.parameters()).device
    for index, example in enumerate(examples):
        prompt = format_prompt(example.question, tokenizer)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_prompt_tokens).to(device)
        generated = model.generate(
            **encoded, do_sample=False, max_new_tokens=config.eval_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True
        )
        completion = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = extract_final_answer(completion)
        is_correct = predicted == example.answer
        correct += int(is_correct)
        append_jsonl(output, {
            "id": example.uid, "source": example.source, "question": example.question,
            "gold": example.answer, "predicted": predicted, "correct": is_correct, "completion": completion,
        })
        print(f"evaluation {index + 1}/{len(examples)} correct={correct}")
    return {"samples": len(examples), "correct": correct, "accuracy": correct / max(len(examples), 1), "elapsed_seconds": time.time() - started}


def evaluate_checkpoint(method: str, config: ExperimentConfig, output_dir: str | Path, adapter_path: str | None) -> None:
    output = Path(output_dir) / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(adapter_path or config.model_name)
    model = load_evaluation_model(config, adapter_path)
    gsm8k = evaluate_examples(
        model, tokenizer, load_gsm8k("test", config.eval_gsm8k_samples, config.seed),
        config, output / "gsm8k_predictions.jsonl"
    )
    math500 = evaluate_examples(
        model, tokenizer, load_math500(config.eval_math500_samples, config.seed),
        config, output / "math500_predictions.jsonl"
    )
    write_json(output / "metrics.json", {"method": method, "gsm8k": gsm8k, "math500": math500})

