from __future__ import annotations

import torch

from pact_grpo.radical_equations import (
    candidate_roots, exact_roots, exact_solution, make_radical_splits,
    parse_solution_set, root_satisfies_original, sound_solution,
)
from pact_grpo.radical_trainer import preserve_soundness


def test_oracle_filters_extraneous_root():
    # Squaring sqrt(5x + 55) = x + 1 gives x=9 and x=-6; only 9 works.
    assert exact_roots(5, 55, 1, 1) == frozenset({9})
    assert root_satisfies_original(5, 55, 1, 1, 9)
    assert not root_satisfies_original(5, 55, 1, 1, -6)


def test_disjoint_stratified_splits_and_solution_grading():
    train, validation, test = make_radical_splits()
    cases = train + validation + test
    assert len(cases) == 800
    assert len({(case.k, case.a, case.m, case.b) for case in cases}) == len(cases)
    for case in cases:
        assert exact_roots(case.k, case.a, case.m, case.b) == case.roots
        assert exact_solution(f"\\boxed{{{case.example.answer}}}", case)
        assert "candidate roots" in case.example.question.lower()
        candidates = candidate_roots(case.k, case.a, case.m, case.b)
        assert candidates is not None and all(str(root) in case.example.question for root in candidates)
    assert {len(case.roots) for case in cases} == {0, 1, 2}
    singleton = next(case for case in cases if len(case.roots) == 1)
    assert not sound_solution("\\boxed{none}", singleton)
    assert not exact_solution("\\boxed{none}", singleton)
    assert parse_solution_set("\\boxed{-1,3}") == frozenset({-1, 3})


def test_projection_preserves_soundness_direction():
    task = torch.tensor([1.0, -2.0])
    soundness = torch.tensor([0.0, 1.0])
    adjusted, applied, margin = preserve_soundness(task, soundness)
    assert applied and margin < 0
    assert float(torch.dot(adjusted, soundness)) >= -1e-7
    assert torch.allclose(adjusted, torch.tensor([1.0, 0.0]))
    adjusted, applied, _ = preserve_soundness(torch.zeros(2), soundness, min_progress=0.1)
    assert applied
    assert torch.allclose(adjusted, torch.tensor([0.0, 0.1]))
