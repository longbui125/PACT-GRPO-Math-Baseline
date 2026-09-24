"""VS Code entry point: old-skill SFT, then four RL transfer methods."""

import gc
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))

import torch
from evaluate import evaluate
from pact_grpo.config import ExperimentConfig
from pact_grpo.modeling import load_evaluation_model, load_tokenizer
from pact_grpo.radical_sft import train_candidate_warm_start
from pact_grpo.radical_trainer import METHODS, train_radical_method
from pact_grpo.transfer_data import make_transfer_splits

OPTIONS = json.loads((ROOT / "configs" / "transfer.json").read_text(encoding="utf-8"))
OUTPUT = ROOT / "outputs" / "transfer_v1"


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_initial(config, adapter: Path, cases, destination: Path) -> None:
    if destination.exists() and (destination.parent / "metrics.json").exists():
        return
    tokenizer = load_tokenizer(str(adapter))
    model = load_evaluation_model(config, str(adapter))
    evaluate(model, tokenizer, cases, config, destination)
    del model, tokenizer
    release_memory()


def run_seed(seed: int) -> None:
    config = replace(ExperimentConfig.from_json(ROOT / "configs" / "radical.json"),
                     seed=seed, max_steps=int(OPTIONS["rl_groups"]))
    splits = make_transfer_splits(seed)
    output = OUTPUT / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    families = {name: getattr(splits, name) for name in (
        "easy_train", "easy_validation", "easy_test",
        "hard_train", "hard_validation", "hard_test")}
    manifest = {"seed": seed, "families": {
        name: {"count": len(cases), "ids_sha256": hashlib.sha256(
            "\n".join(case.example.uid for case in cases).encode()
        ).hexdigest()} for name, cases in families.items()},
        "source": "deterministic synthetic equations; see transfer_data.py"}
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sft_folder = output / "initial_adapter"
    adapter = sft_folder / "adapter"
    previous_summary = sft_folder / "summary.json"
    if previous_summary.exists():
        previous = json.loads(previous_summary.read_text(encoding="utf-8"))
        if previous.get("epochs") != int(OPTIONS["sft_epochs"]):
            raise RuntimeError(
                "The saved initial adapter used a different SFT epoch count. "
                "Use a new output directory for the revised experiment."
            )
    if not (adapter.exists() and (sft_folder / "summary.json").exists()):
        train_candidate_warm_start(config, sft_folder, splits.easy_train,
                                   epochs=int(OPTIONS["sft_epochs"]))
        release_memory()

    validation_zero = [case for case in splits.easy_validation if not case.roots]
    validation_file = output / "initial" / "validation_easy_zero" / "predictions.jsonl"
    evaluate_initial(config, adapter, validation_zero, validation_file)
    validation = json.loads((validation_file.parent / "metrics.json").read_text())
    if validation["exact_set_accuracy"] < OPTIONS["min_easy_zero_validation_accuracy"]:
        raise RuntimeError(
            f"Initial SFT rejects only {validation['exact_set_accuracy']:.1%} of "
            "easy validation no-solution cases; retention is not yet testable. "
            "Improve the old-skill SFT before running RL."
        )
    hard_validation_file = output / "initial" / "validation_hard" / "predictions.jsonl"
    evaluate_initial(config, adapter, splits.hard_validation, hard_validation_file)
    hard_validation = json.loads((hard_validation_file.parent / "metrics.json").read_text())
    print(f"[{seed}] initial B-validation accuracy: "
          f"{hard_validation['exact_set_accuracy']:.1%}", flush=True)

    train_zero = [case for case in splits.easy_train if not case.roots]
    calibration_file = output / "initial" / "anchor_calibration" / "predictions.jsonl"
    evaluate_initial(config, adapter, train_zero, calibration_file)
    calibration = [json.loads(line) for line in calibration_file.read_text(
        encoding="utf-8").splitlines()]
    mastered_ids = {row["id"] for row in calibration if row["correct"]}
    anchors = [case for case in train_zero if case.example.uid in mastered_ids]
    if len(anchors) < OPTIONS["min_calibrated_anchors"]:
        raise RuntimeError(f"Only {len(anchors)} mastered anchors; need more old-skill SFT")
    (output / "calibrated_anchor_ids.json").write_text(json.dumps(
        [case.example.uid for case in anchors], indent=2), encoding="utf-8")

    for method in OPTIONS["methods"]:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method}")
        destination = output / method
        if (destination / "summary.json").exists() and (destination / "adapter").exists():
            print(f"[{seed}/{method}] completed; skipping", flush=True)
            continue
        print(f"[{seed}/{method}] training", flush=True)
        train_radical_method(
            method, config, destination, adapter,
            case_focus="mixed_one_root", anchor_strategy="protect_zero_root",
            training_cases=splits.hard_train, anchor_cases=anchors,
            checkpoint_every=int(OPTIONS["checkpoint_every"]),
        )
        release_memory()


if __name__ == "__main__":
    for run_seed_value in OPTIONS["seeds"]:
        run_seed(int(run_seed_value))
