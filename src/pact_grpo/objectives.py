"""GRPO surrogate objectives shared by all update-direction methods."""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, *, normalize_std: bool) -> torch.Tensor:
    centered = rewards - rewards.mean()
    if not normalize_std:
        return centered
    std = rewards.std(unbiased=False)
    if std.item() < 1e-8:
        return torch.zeros_like(centered)
    return centered / (std + 1e-8)


def rollout_surrogates(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float,
    dr_grpo: bool,
    max_completion_tokens: int,
) -> torch.Tensor:
    """One clipped policy objective per rollout; mean is the training objective.

    GRPO averages over completion tokens. Dr. GRPO uses a fixed denominator
    instead, avoiding response-length weighting.
    """
    if new_logprobs.shape != old_logprobs.shape or mask.shape != new_logprobs.shape:
        raise ValueError("Log probabilities and token mask must have identical shapes")
    if advantages.shape != (new_logprobs.shape[0],):
        raise ValueError("One advantage is required per rollout")
    ratio = (new_logprobs - old_logprobs).clamp(-20, 20).exp()
    advantage = advantages[:, None]
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage
    token_objective = torch.minimum(unclipped, clipped) * mask
    if dr_grpo:
        denominator = float(max_completion_tokens)
    else:
        denominator = mask.sum(dim=1).clamp_min(1)
    return token_objective.sum(dim=1) / denominator
