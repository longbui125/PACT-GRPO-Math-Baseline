"""VS Code entry point: exact-set evaluation of radical equations."""

import gc
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))

import torch
import pact_grpo.data as data
from pact_grpo.config import ExperimentConfig
from pact_grpo.modeling import load_evaluation_model, load_tokenizer
from pact_grpo.data import ANSWER_PREFIX
from pact_grpo.radical_equations import exact_solution, parse_solution_set, sound_solution
from pact_grpo.transfer_data import make_transfer_splits

OPTIONS = json.loads((ROOT / "configs" / "transfer.json").read_text(encoding="utf-8"))
OUTPUT = ROOT / "outputs" / "transfer_v1"


@torch.no_grad()
def evaluate(model, tokenizer, cases, config, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    for index, case in enumerate(cases, 1):
        prompt = data.format_prompt(case.example.question, tokenizer)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=config.max_prompt_tokens).to(next(model.parameters()).device)
        generated = model.generate(
            **encoded, do_sample=False, max_new_tokens=config.eval_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True,
        )
        completion = ANSWER_PREFIX + tokenizer.decode(
            generated[0, encoded.input_ids.shape[1]:], skip_special_tokens=True
        )
        roots = parse_solution_set(completion)
        row = {
            "id": case.example.uid, "question": case.example.question,
            "gold": sorted(case.roots), "predicted": None if roots is None else sorted(roots),
            "correct": exact_solution(completion, case), "sound": sound_solution(completion, case),
            "completion": completion, "generated_tokens": int(generated.shape[1] - encoded.input_ids.shape[1]),
        }
        rows.append(row)
        if index == 1 or index % 10 == 0 or index == len(cases):
            print(f"evaluation {index}/{len(cases)} exact={sum(r['correct'] for r in rows)}", flush=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metrics = {
        "samples": len(rows), "correct": sum(row["correct"] for row in rows),
        "exact_set_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "soundness_rate": sum(row["sound"] for row in rows) / len(rows),
        "parse_rate": sum(row["predicted"] is not None for row in rows) / len(rows),
        "generated_tokens": sum(row["generated_tokens"] for row in rows),
        "seconds": time.time() - started,
        "by_root_count": {str(count): {
            "correct": sum(row["correct"] for row in rows if len(row["gold"]) == count),
            "samples": sum(len(row["gold"]) == count for row in rows),
        } for count in (0, 1, 2)},
    }
    (output.parent / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def evaluate_checkpoint(config, adapter: Path, cases, destination: Path) -> None:
    if destination.exists() and (destination.parent / "metrics.json").exists():
        print(f"Already evaluated: {destination}", flush=True)
        return
    if not adapter.exists():
        raise FileNotFoundError(f"Run train.py first: {adapter}")
    tokenizer = load_tokenizer(str(adapter))
    model = load_evaluation_model(config, str(adapter))
    evaluate(model, tokenizer, cases, config, destination)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_seed(seed: int) -> None:
    config = replace(ExperimentConfig.from_json(ROOT / "configs" / "radical.json"), seed=seed)
    splits = make_transfer_splits(seed)
    base = OUTPUT / f"seed_{seed}"
    easy_val_zero = [case for case in splits.easy_validation if not case.roots]
    test_splits = {"test_easy": splits.easy_test, "test_hard": splits.hard_test}
    validation_splits = {
        "validation_easy_zero": easy_val_zero,
        "validation_hard": splits.hard_validation,
    }
    for method in ("initial", *OPTIONS["methods"]):
        adapter = (base / "initial_adapter" / "adapter" if method == "initial"
                   else base / method / "adapter")
        for split_name, cases in test_splits.items():
            evaluate_checkpoint(config, adapter, cases,
                                base / method / split_name / "predictions.jsonl")
        for split_name, cases in validation_splits.items():
            evaluate_checkpoint(config, adapter, cases,
                                base / method / split_name / "predictions.jsonl")
        if method == "initial":
            continue
        summary = json.loads((base / method / "summary.json").read_text(encoding="utf-8"))
        for step in summary["checkpoints"]:
            if step == summary["target_groups"]:
                continue
            checkpoint = base / method / "checkpoints" / f"step_{step}"
            for split_name, cases in validation_splits.items():
                evaluate_checkpoint(config, checkpoint / "adapter", cases,
                                    checkpoint / split_name / "predictions.jsonl")


if __name__ == "__main__":
    for run_seed_value in OPTIONS["seeds"]:
        run_seed(int(run_seed_value))
