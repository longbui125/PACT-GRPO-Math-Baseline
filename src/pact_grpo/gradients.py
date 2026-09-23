from __future__ import annotations

from dataclasses import dataclass

import torch

from .modeling import parameter_slices


def rollout_gradients(objectives: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    rows = []
    for index in range(len(objectives)):
        gradients = torch.autograd.grad(
            objectives[index], parameters, retain_graph=index < len(objectives) - 1, allow_unused=True
        )
        pieces = [
            (torch.zeros_like(parameter) if gradient is None else gradient).detach().float().reshape(-1)
            for gradient, parameter in zip(gradients, parameters)
        ]
        rows.append(torch.cat(pieces))
    return torch.stack(rows)


def assign_ascent_gradient(direction: torch.Tensor, parameters: list[torch.nn.Parameter]) -> None:
    for parameter, section in parameter_slices(parameters):
        parameter.grad = -direction[section].view_as(parameter).to(parameter.dtype)


@dataclass(frozen=True)
class ViolationStats:
    overall: float
    positive: float
    negative: float
    magnitude: float
    minimum_margin: float


def violation_stats(rollout_directions: torch.Tensor, advantages: torch.Tensor, direction: torch.Tensor, tolerance: float) -> ViolationStats:
    """A signed rollout gradient conflicts when its dot product is negative."""
    active = (advantages.abs() > tolerance) & (
        rollout_directions.norm(dim=1) > tolerance
    )
    if not bool(active.any()):
        return ViolationStats(0.0, 0.0, 0.0, 0.0, 0.0)
    active_advantages = advantages[active]
    margins = rollout_directions[active] @ direction
    violated = margins < -tolerance

    def rate(mask: torch.Tensor) -> float:
        return float(violated[mask].float().mean()) if bool(mask.any()) else 0.0

    return ViolationStats(
        float(violated.float().mean()), rate(active_advantages > 0), rate(active_advantages < 0),
        float(torch.relu(-margins).mean()), float(margins.min())
    )
