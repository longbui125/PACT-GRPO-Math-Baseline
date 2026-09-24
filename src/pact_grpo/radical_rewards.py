"""Verifier rewards for selecting valid roots from supplied candidates."""

from __future__ import annotations

from .radical_equations import RadicalCase, candidate_roots, parse_solution_set


def candidate_overlap(predicted: frozenset[int] | None,
                      case: RadicalCase) -> float:
    """Jaccard overlap for nonempty valid sets; invalid candidates earn nothing."""
    candidates = set(candidate_roots(case.k, case.a, case.m, case.b) or ())
    if predicted is None or not predicted or not case.roots:
        return 0.0
    if not predicted.issubset(candidates):
        return 0.0
    return len(predicted & case.roots) / len(predicted | case.roots)


def verified_reward(text: str, case: RadicalCase, *,
                    correctness_weight: float = 1.0,
                    format_weight: float = 0.05,
                    candidate_overlap_weight: float = 0.5) -> float:
    predicted = parse_solution_set(text)
    candidates = set(candidate_roots(case.k, case.a, case.m, case.b) or ())
    if predicted is None or not predicted.issubset(candidates):
        return 0.0
    exact = predicted == case.roots
    return (
        format_weight
        + candidate_overlap_weight * candidate_overlap(predicted, case)
        + correctness_weight * float(exact)
    )
