from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProjectionResult:
    direction: torch.Tensor
    applied: bool
    iterations: int
    norm_ratio: float
    max_violation: float


def project_halfspaces(direction: torch.Tensor, constraints: torch.Tensor, tolerance: float, max_iterations: int, ridge: float) -> ProjectionResult:
    """Euclidean projection onto the intersection constraints @ d >= 0."""
    if constraints.numel() == 0:
        return ProjectionResult(direction, False, 0, 1.0, 0.0)
    initial = constraints @ direction
    if bool((initial >= -tolerance).all()):
        return ProjectionResult(direction, False, 0, 1.0, 0.0)
    gram = constraints @ constraints.T
    gram += ridge * torch.eye(len(constraints), device=gram.device, dtype=gram.dtype)
    step_size = 1.0 / float(torch.linalg.eigvalsh(gram).max().clamp_min(ridge))
    dual = torch.zeros(len(constraints), device=gram.device, dtype=gram.dtype)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        updated = torch.clamp(dual - step_size * (gram @ dual + initial), min=0.0)
        if float((updated - dual).abs().max()) <= tolerance:
            dual = updated
            break
        dual = updated
    projected = direction + constraints.T @ dual
    margins = constraints @ projected
    ratio = float(projected.norm() / direction.norm().clamp_min(1e-12))
    return ProjectionResult(projected, True, iterations, ratio, float(torch.relu(-margins).max()))


def pcgrad(vectors: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if len(vectors) <= 1:
        return vectors.mean(0)
    modified = vectors.clone()
    for index in range(len(vectors)):
        for other in torch.randperm(len(vectors), generator=generator).tolist():
            if other == index:
                continue
            reference = vectors[other]
            dot = torch.dot(modified[index], reference)
            if float(dot) < 0:
                modified[index] -= dot / torch.dot(reference, reference).clamp_min(1e-12) * reference
    return modified.mean(0)

