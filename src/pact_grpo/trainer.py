"""One controlled GRPO training loop with interchangeable update directions."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import torch

from .config import ExperimentConfig
from .data import load_gsm8k
from .gradients import assign_ascent_gradient, rollout_gradients, violation_stats
from .modeling import load_tokenizer, load_trainable_model, trainable_parameters
from .objectives import group_advantages, rollout_surrogates
from .projections import pcgrad, project_halfspaces
from .rollouts import completion_logprobs, generate_group
from .utils import append_jsonl, mean, set_seed, write_json

Method = Literal["grpo", "dr_grpo", "pcgrad_grpo", "pact_grpo"]
METHODS = frozenset({"grpo", "dr_grpo", "pcgrad_grpo", "pact_grpo"})


def select_direction(
    method: Method,
    rollout_directions: torch.Tensor,
    tolerance: float,
    iterations: int,
    ridge: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, bool, int, float, float]:
    """Return direction, changed?, iterations, norm ratio, max violation."""
    grpo = rollout_directions.mean(dim=0)
    if method == "pcgrad_grpo":
        chosen = pcgrad(rollout_directions, generator)
        changed = not torch.allclose(chosen, grpo)
        ratio = float(chosen.norm() / grpo.norm().clamp_min(1e-12))
        return chosen, changed, 0, ratio, 0.0
    if method == "pact_grpo":
        active = rollout_directions.norm(dim=1) > tolerance
        projected = project_halfspaces(
            grpo, rollout_directions[active], tolerance, iterations, ridge
        )
        return (
            projected.direction, projected.applied, projected.iterations,
            projected.norm_ratio, projected.max_violation,
        )
    return grpo, False, 0, 1.0, 0.0


def train(method: Method, config: ExperimentConfig, output_dir: str | Path) -> None:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose one of {sorted(METHODS)}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "train_metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    config.save(output / "config.json", {"method": method})
    set_seed(config.seed)

    tokenizer = load_tokenizer(config.model_name)
    model = load_trainable_model(config)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    examples = load_gsm8k("train", config.train_samples, config.seed)
    if not examples:
        raise ValueError("Training split is empty")
    random_order = torch.Generator(device="cpu").manual_seed(config.seed)
    indices = torch.randperm(len(examples), generator=random_order).tolist()

    rewards, rsvr_before, rsvr_after, norm_ratios = [], [], [], []
    projected_steps, started = 0, time.time()

    for step in range(config.max_steps):
        example = examples[indices[step % len(indices)]]
        group = generate_group(model, tokenizer, example, config)
        advantage = group_advantages(
            group.rewards, normalize_std=method != "dr_grpo"
        )
        model.eval()
        with torch.no_grad():
            old_logprobs = completion_logprobs(model, group).detach()
        model.train()

        for epoch in range(config.policy_epochs):
            current_logprobs = completion_logprobs(model, group)
            surrogates = rollout_surrogates(
                current_logprobs, old_logprobs, group.completion_mask,
                advantage, clip_epsilon=config.clip_epsilon,
                dr_grpo=method == "dr_grpo",
                max_completion_tokens=config.max_completion_tokens,
            )
            per_rollout = rollout_gradients(surrogates, parameters)
            if bool(advantage.abs().max() > 0) and not bool(per_rollout.abs().max() > 0):
                raise RuntimeError(
                    "Nonzero advantages produced zero LoRA gradients; check "
                    "gradient checkpointing and completion masks."
                )
            grpo_direction = per_rollout.mean(dim=0)
            before = violation_stats(
                per_rollout, advantage, grpo_direction, config.projection_tolerance
            )
            direction, applied, iterations, norm_ratio, max_violation = select_direction(
                method, per_rollout, config.projection_tolerance,
                config.projection_iterations, config.projection_ridge, random_order
            )
            after = violation_stats(
                per_rollout, advantage, direction, config.projection_tolerance
            )

            optimizer.zero_grad(set_to_none=True)
            assign_ascent_gradient(direction, parameters)
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            optimizer.step()

            projected_steps += int(applied)
            rsvr_before.append(before.overall)
            rsvr_after.append(after.overall)
            norm_ratios.append(norm_ratio)
            append_jsonl(metrics_path, {
                "step": step + 1, "epoch": epoch + 1, "method": method,
                "example_id": example.uid,
                "reward_mean": float(group.rewards.mean()),
                "reward_std": float(group.rewards.std(unbiased=False)),
                "correct_rate": mean(item.correctness for item in group.breakdowns),
                "rsvr_before": before.overall,
                "rsvr_positive_before": before.positive,
                "rsvr_negative_before": before.negative,
                "violation_magnitude_before": before.magnitude,
                "rsvr_after": after.overall,
                "rsvr_positive_after": after.positive,
                "rsvr_negative_after": after.negative,
                "violation_magnitude_after": after.magnitude,
                "projection_applied": applied,
                "projection_iterations": iterations,
                "projection_norm_ratio": norm_ratio,
                "max_post_violation": max_violation,
                "direction_norm": float(direction.norm()),
                "elapsed_seconds": time.time() - started,
            })

        rewards.append(float(group.rewards.mean()))
        print(
            f"[{method}] group={step + 1}/{config.max_steps} "
            f"reward={rewards[-1]:.3f} RSVR={rsvr_after[-1]:.3f}"
        )
        if (step + 1) % config.save_every == 0:
            checkpoint = output / "checkpoints" / f"step-{step + 1}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)

    adapter = output / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    write_json(output / "summary.json", {
        "method": method,
        "groups": config.max_steps,
        "optimizer_updates": config.max_steps * config.policy_epochs,
        "mean_training_reward": mean(rewards),
        "mean_rsvr_before": mean(rsvr_before),
        "mean_post_update_rsvr": mean(rsvr_after),
        "mean_projection_norm_ratio": mean(norm_ratios),
        "projection_step_rate": projected_steps / max(len(rsvr_after), 1),
        "elapsed_seconds": time.time() - started,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "constraint_space": "raw LoRA surrogate gradient before AdamW",
    })
