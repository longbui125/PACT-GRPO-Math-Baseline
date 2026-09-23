"""Reproduce the 6 GB GPU pilot in VS Code using the existing tf_gpu interpreter.

Set MODE below to 'train' or 'evaluate' to rerun. 'summary' only reads results.
Training replaces the selected methods' earlier output, so archive results first.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = Path(os.getenv("DATN_CACHE_ROOT", r"C:\Users\slywi\Documents\Codex\2026-09-11\h-y"))
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf_cache"))
sys.path.insert(0, str(ROOT / "src"))

import torch
from pact_grpo.config import ExperimentConfig
from pact_grpo.evaluation import evaluate_checkpoint
from pact_grpo.trainer import train
import pact_grpo.data as data

MODE = "summary"  # summary, train, evaluate
SCALE = "verified_pilot"
METHODS = ("grpo", "pact_grpo")

# Identical prompt for training and evaluation of all methods in this pilot.
data.SYSTEM_PROMPT = (
    "Solve the math problem. Use at most two short calculation lines, then "
    "write the final answer as \\boxed{number}. Do not add other text."
)


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    config = ExperimentConfig.from_json(ROOT / "configs" / f"{SCALE}.json")
    output = ROOT / "outputs" / SCALE
    if MODE == "train":
        for method in METHODS:
            train(method, config, output / method)
            release_memory()
    elif MODE == "evaluate":
        for method in ("base", *METHODS):
            adapter = None if method == "base" else output / method / "adapter"
            evaluate_checkpoint(method, config, output / method, str(adapter) if adapter else None)
            release_memory()
    elif MODE == "summary":
        for method in ("base", *METHODS):
            folder = output / method
            eval_file = folder / "evaluation" / "metrics.json"
            train_file = folder / "summary.json"
            evaluation = json.loads(eval_file.read_text(encoding="utf-8")) if eval_file.exists() else {}
            training = json.loads(train_file.read_text(encoding="utf-8")) if train_file.exists() else {}
            print(method, evaluation.get("gsm8k"), evaluation.get("math500"),
                  "post-RSVR:", training.get("mean_post_update_rsvr"))
    else:
        raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    main()
