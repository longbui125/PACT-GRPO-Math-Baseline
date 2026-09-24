"""GRPO variants and PACT for checking candidate radical roots."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import format_prompt
from .gradients import assign_ascent_gradient
from .modeling import load_tokenizer, trainable_parameters
from .objectives import group_advantages, rollout_surrogates
from .radical_equations import (
    RadicalCase, candidate_roots, exact_solution, make_radical_splits,
    parse_solution_set, sound_solution,
)
from .radical_modeling import load_trainable_adapter_model
from .radical_rewards import verified_reward
from .rollouts import completion_logprobs, generate_group
from .utils import append_jsonl, set_seed, write_json

METHODS = ("grpo", "dr_grpo", "dapo", "pact_grpo")
DAPO_CLIP_HIGH = 0.28
DAPO_RETRIES = 8
DAPO_OVERLONG_CACHE = 4
PACT_ANCHOR_PROGRESS = 0.0


def _flat_gradient(objective: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    gradients = torch.autograd.grad(objective, parameters, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(parameter) if gradient is None else gradient)
        .detach().float().reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ])


def preserve_soundness(direction: torch.Tensor, anchor_direction: torch.Tensor,
                       min_progress: float = 0.0) -> tuple[torch.Tensor, bool, float]:
    """Minimum Euclidean projection onto g_anchor · d >= c ||g_anchor||²."""
    if min_progress < 0:
        raise ValueError("min_progress must be nonnegative")
    margin = torch.dot(direction, anchor_direction)
    norm_squared = torch.dot(anchor_direction, anchor_direction)
    target = min_progress * norm_squared
    if float(norm_squared) <= 1e-16 or float(margin) >= float(target):
        return direction, False, float(margin)
    return direction + (target - margin) / norm_squared * anchor_direction, True, float(margin)


def _answer_logprob(model, tokenizer, case: RadicalCase, answer: str) -> torch.Tensor:
    prompt_ids = tokenizer(
        format_prompt(case.example.question, tokenizer), add_special_tokens=False,
    ).input_ids
    response_ids = tokenizer(
        f"{answer}}}" + tokenizer.eos_token, add_special_tokens=False,
    ).input_ids
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids + response_ids], device=device)
    logits = model(input_ids=ids, use_cache=False).logits
    response_logits = logits[:, len(prompt_ids) - 1:-1, :]
    targets = torch.tensor(response_ids, device=device)
    return F.log_softmax(response_logits.float(), dim=-1)[0].gather(
        1, targets[:, None],
    ).mean()


def _anchor_gradient(model, tokenizer, case: RadicalCase,
                     parameters: list[torch.nn.Parameter], kind: str) -> torch.Tensor:
    candidates = candidate_roots(case.k, case.a, case.m, case.b)
    if kind == "soundness":
        if candidates is None or len(candidates) < 2 or set(candidates) == set(case.roots):
            raise ValueError("Soundness anchor must contain an extraneous root")
        invalid_answer = ",".join(str(value) for value in candidates)
    elif kind == "zero_root_soundness":
        if candidates is None or not candidates or case.roots:
            raise ValueError("Zero-root anchor needs extraneous candidates only")
        invalid_answer = str(candidates[0])
    elif kind == "completeness":
        if len(case.roots) != 2:
            raise ValueError("Completeness anchor needs two valid roots")
        invalid_answer = str(min(case.roots))
    else:
        raise ValueError(f"Unknown anchor kind: {kind}")
    objective = (_answer_logprob(model, tokenizer, case, case.example.answer)
                 - _answer_logprob(model, tokenizer, case, invalid_answer))
    return _flat_gradient(objective, parameters)


def _case_for_index(hard: list[RadicalCase], other: list[RadicalCase],
                    index: int) -> RadicalCase:
    return hard[(index // 2) % len(hard)] if index % 2 == 0 else other[(index // 2) % len(other)]


def _rewards(config: ExperimentConfig, texts: list[str], case: RadicalCase,
             completion_lengths: torch.Tensor, method: str) -> torch.Tensor:
    rewards = torch.tensor(
        [verified_reward(
            text, case,
            correctness_weight=config.correctness_reward_weight,
            format_weight=config.format_reward_weight,
            candidate_overlap_weight=config.candidate_overlap_reward_weight,
        ) for text in texts],
        dtype=torch.float32, device=completion_lengths.device,
    )
    if method == "dapo":
        if config.max_completion_tokens <= DAPO_OVERLONG_CACHE:
            raise ValueError("DAPO max_completion_tokens must exceed its overlong cache")
        start = config.max_completion_tokens - DAPO_OVERLONG_CACHE
        rewards -= 0.1 * ((completion_lengths.float() - start) / DAPO_OVERLONG_CACHE).clamp(0, 1)
    return rewards


def train_radical_method(method: str, config: ExperimentConfig,
                        output_dir: str | Path, warm_start_adapter: str | Path,
                        *, case_focus: str = "balanced",
                        anchor_strategy: str = "mixed",
                        training_cases: list[RadicalCase] | None = None,
                        anchor_cases: list[RadicalCase] | None = None,
                        checkpoint_every: int = 0) -> None:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    if case_focus not in {"balanced", "mixed_one_root"}:
        raise ValueError(f"Unknown case focus: {case_focus}")
    if anchor_strategy not in {"mixed", "protect_zero_root"}:
        raise ValueError(f"Unknown anchor strategy: {anchor_strategy}")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_file = output / "train_metrics.jsonl"
    metrics_file.unlink(missing_ok=True)
    config.save(output / "config.json", {
        "method": method,
        "application": "radical_candidate_verification",
        "initialization": "provided_sft_adapter",
        "reward": "exact_set_plus_candidate_overlap",
        "case_focus": case_focus,
        "anchor_strategy": anchor_strategy,
        "training_cases": len(training_cases) if training_cases is not None else None,
        "anchor_cases": len(anchor_cases) if anchor_cases is not None else None,
        "constraint": "anchor_contrastive_progress" if method == "pact_grpo" else None,
        "pact_anchor_progress": PACT_ANCHOR_PROGRESS if method == "pact_grpo" else None,
        "dapo_clip_high": DAPO_CLIP_HIGH if method == "dapo" else None,
        "dapo_dynamic_sampling_max_attempts": DAPO_RETRIES if method == "dapo" else None,
        "dapo_soft_overlong_cache_tokens": DAPO_OVERLONG_CACHE if method == "dapo" else None,
    })
    set_seed(config.seed)
    if training_cases is None:
        train, _, _ = make_radical_splits(config.seed)
    else:
        train = training_cases
    protected = train if anchor_cases is None else anchor_cases
    soundness_anchors = [case for case in protected if len(case.roots) == 1
                         and len(candidate_roots(case.k, case.a, case.m, case.b) or ()) == 2]
    zero_root_anchors = [case for case in protected if len(case.roots) == 0]
    completeness_anchors = [case for case in protected if len(case.roots) == 2]
    hard = [case for case in train if len(case.roots) == 2]
    other = [case for case in train if len(case.roots) != 2]
    focus_pool = [case for case in train if len(case.roots) == 1
                  and len(candidate_roots(case.k, case.a, case.m, case.b) or ()) == 2]
    if case_focus == "balanced" and not (hard and other):
        raise ValueError("Balanced training requires two-root and other cases")
    if case_focus == "mixed_one_root" and not focus_pool:
        raise ValueError("Focused training requires one-valid-one-extraneous cases")
    if method == "pact_grpo" and anchor_strategy == "mixed" and not (
        soundness_anchors and completeness_anchors
    ):
        raise ValueError("Mixed PACT anchors require soundness and completeness cases")
    if method == "pact_grpo" and anchor_strategy == "protect_zero_root" and not zero_root_anchors:
        raise ValueError("PACT protection requires zero-root anchors")
    rng = torch.Generator().manual_seed(config.seed)
    hard = [hard[index] for index in torch.randperm(len(hard), generator=rng).tolist()]
    other = [other[index] for index in torch.randperm(len(other), generator=rng).tolist()]
    focus_cases = [focus_pool[index] for index in
                   torch.randperm(len(focus_pool), generator=rng).tolist()]

    tokenizer = load_tokenizer(str(warm_start_adapter))
    model = load_trainable_adapter_model(config, str(warm_start_adapter))
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                   weight_decay=config.weight_decay)
    started = time.time()
    rollout_tokens = 0
    sampled_groups = 0
    projected_steps = 0
    zero_task_updates = 0
    optimizer_updates = 0
    completed_groups = 0
    early_stop_reason = None
    skipped_groups = 0
    informative_groups = 0
    checkpoints: list[int] = []

    for step in range(config.max_steps):
        max_attempts = DAPO_RETRIES if method == "dapo" else 1
        for attempt in range(max_attempts):
            case = (focus_cases[(step + attempt) % len(focus_cases)]
                    if case_focus == "mixed_one_root"
                    else _case_for_index(hard, other, step + attempt))
            group = generate_group(model, tokenizer, case.example, config)
            sampled_groups += 1
            rollout_tokens += int(group.completion_mask.sum())
            exact_flags = [float(exact_solution(text, case)) for text in group.texts]
            sound_flags = [float(sound_solution(text, case)) for text in group.texts]
            format_flags = [float(parse_solution_set(text) is not None) for text in group.texts]
            lengths = group.completion_mask.sum(dim=1)
            task_rewards = _rewards(config, group.texts, case, lengths, method)
            if method != "dapo" or float(task_rewards.max() - task_rewards.min()) > 1e-6:
                break
        else:
            skipped_groups += 1
            append_jsonl(metrics_file, {
                "step": step + 1, "skipped": True,
                "sampling_attempts": max_attempts,
                "rollout_tokens": rollout_tokens,
                "elapsed_seconds": time.time() - started,
                "reason": "No reward variance after DAPO retries",
            })
            print(f"[dapo] step {step + 1}: no reward variance; skipped", flush=True)
            continue

        informative_groups += int(float(task_rewards.max() - task_rewards.min()) > 1e-6)
        task_advantages = group_advantages(
            task_rewards, normalize_std=method != "dr_grpo",
        )
        model.eval()
        with torch.no_grad():
            old_logprobs = completion_logprobs(model, group).detach()

        if method == "pact_grpo":
            if anchor_strategy == "protect_zero_root":
                anchor_type = "zero_root_soundness"
                anchor = zero_root_anchors[step % len(zero_root_anchors)]
            else:
                anchor_type = "soundness" if step % 2 == 0 else "completeness"
                anchors = soundness_anchors if anchor_type == "soundness" else completeness_anchors
                anchor = anchors[(step // 2) % len(anchors)]
        else:
            anchor_type = None

        for epoch in range(config.policy_epochs):
            model.train()
            current_logprobs = completion_logprobs(model, group)
            rollout_values = rollout_surrogates(
                current_logprobs, old_logprobs, group.completion_mask, task_advantages,
                clip_epsilon=config.clip_epsilon, dr_grpo=method == "dr_grpo",
                max_completion_tokens=config.max_completion_tokens,
                clip_epsilon_high=DAPO_CLIP_HIGH if method == "dapo" else None,
            )
            if method == "dapo":
                objective = (rollout_values * lengths).sum() / lengths.sum().clamp_min(1)
            else:
                objective = rollout_values.mean()
            task_direction = _flat_gradient(objective, parameters)
            del current_logprobs, rollout_values, objective

            task_norm = float(task_direction.norm())
            if task_norm <= 1e-12:
                # A homogeneous group supplies no policy-gradient signal.
                # In particular, PACT must not turn a zero GRPO update into
                # supervised anchor training.
                direction, applied, margin = task_direction, False, 0.0
                zero_task_updates += 1
            elif method == "pact_grpo":
                anchor_direction = _anchor_gradient(model, tokenizer, anchor, parameters,
                                                    anchor_type)
                direction, applied, margin = preserve_soundness(
                    task_direction, anchor_direction, min_progress=PACT_ANCHOR_PROGRESS,
                )
            else:
                direction, applied, margin = task_direction, False, 0.0
            projected_steps += int(applied)
            if float(direction.norm()) > 1e-12:
                optimizer.zero_grad(set_to_none=True)
                assign_ascent_gradient(direction, parameters)
                torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
                optimizer.step()
                optimizer_updates += 1

            append_jsonl(metrics_file, {
                "step": step + 1, "policy_epoch": epoch + 1,
                "skipped": False,
                "case": case.example.uid, "sampling_attempts": attempt + 1,
                "exact_rollout_rate": sum(exact_flags) / config.group_size,
                "sound_rollout_rate": sum(sound_flags) / config.group_size,
                "format_rollout_rate": sum(format_flags) / config.group_size,
                "mean_reward": float(task_rewards.mean()),
                "reward_std": float(task_rewards.std(unbiased=False)),
                "projected": applied, "unprojected_anchor_margin": margin,
                "anchor": anchor.example.uid if method == "pact_grpo" else None,
                "anchor_type": anchor_type,
                "task_direction_norm": task_norm,
                "update_direction_norm": float(direction.norm()),
                "rollout_tokens": rollout_tokens, "elapsed_seconds": time.time() - started,
            })
        completed_groups += 1
        if checkpoint_every and (step + 1) % checkpoint_every == 0:
            checkpoint = output / "checkpoints" / f"step_{step + 1}" / "adapter"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            checkpoints.append(step + 1)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == config.max_steps:
            print(f"[{method}] {step + 1}/{config.max_steps} "
                  f"exact={sum(exact_flags) / config.group_size:.2f} "
                  f"reward={float(task_rewards.mean()):.3f} "
                  f"updates={optimizer_updates} sampled_groups={sampled_groups}",
                  flush=True)

    adapter = output / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    write_json(output / "summary.json", {
        "method": method,
        "case_focus": case_focus, "anchor_strategy": anchor_strategy,
        "application": "radical_candidate_verification",
        "groups_used": completed_groups,
        "optimizer_updates": optimizer_updates,
        "target_groups": config.max_steps,
        "early_stopped": early_stop_reason is not None,
        "early_stop_reason": early_stop_reason,
        "sampled_groups": sampled_groups, "rollout_tokens": rollout_tokens,
        "informative_groups": informative_groups, "skipped_groups": skipped_groups,
        "projected_steps": projected_steps, "zero_task_updates": zero_task_updates,
        "checkpoints": checkpoints,
        "elapsed_seconds": time.time() - started,
    })
