from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    model_name: str
    train_samples: int
    max_steps: int
    group_size: int
    max_prompt_tokens: int
    max_completion_tokens: int
    temperature: float
    top_p: float
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    clip_epsilon: float
    policy_epochs: int
    seed: int
    save_every: int
    dtype: str
    gradient_checkpointing: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: list[str]
    format_reward_weight: float
    correctness_reward_weight: float
    projection_tolerance: float
    projection_iterations: int
    projection_ridge: float
    eval_gsm8k_samples: int
    eval_math500_samples: int
    eval_max_new_tokens: int

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        config = cls(**payload)
        if config.group_size < 2 or config.policy_epochs < 1:
            raise ValueError("group_size must be at least 2 and policy_epochs at least 1")
        if not 0 < config.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be between 0 and 1")
        return config

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        payload = asdict(self)
        payload.update(extra or {})
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
