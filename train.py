"""Train one method. Edit METHOD and SCALE, then run this file in VS Code."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pact_grpo.config import ExperimentConfig
from pact_grpo.trainer import train

METHOD = "grpo"  # grpo, dr_grpo, pcgrad_grpo, pact_grpo
SCALE = "pilot"  # pilot or main

if __name__ == "__main__":
    config = ExperimentConfig.from_json(ROOT / "configs" / f"{SCALE}.json")
    train(METHOD, config, ROOT / "outputs" / SCALE / METHOD)
