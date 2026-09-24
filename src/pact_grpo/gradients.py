"""Assign a flat ascent direction to the LoRA parameters."""

from __future__ import annotations

import torch

from .modeling import parameter_slices


def assign_ascent_gradient(direction: torch.Tensor,
                           parameters: list[torch.nn.Parameter]) -> None:
    for parameter, section in parameter_slices(parameters):
        parameter.grad = -direction[section].view_as(parameter).to(parameter.dtype)
