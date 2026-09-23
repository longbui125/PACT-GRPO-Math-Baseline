"""Evaluate a base model or one trained adapter on shared math benchmarks."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pact_grpo.config import ExperimentConfig
from pact_grpo.evaluation import evaluate_checkpoint

METHOD = "base"  # base, grpo, dr_grpo, pcgrad_grpo, pact_grpo
SCALE = "pilot"

if __name__ == "__main__":
    config = ExperimentConfig.from_json(ROOT / "configs" / f"{SCALE}.json")
    run_dir = ROOT / "outputs" / SCALE / METHOD
    adapter = None if METHOD == "base" else run_dir / "adapter"
    if adapter is not None and not adapter.is_dir():
        raise FileNotFoundError(f"Train {METHOD} first: {adapter}")
    evaluate_checkpoint(METHOD, config, run_dir, str(adapter) if adapter else None)
