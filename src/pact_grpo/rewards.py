from __future__ import annotations

from dataclasses import dataclass

from .data import extract_final_answer, normalize_answer


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    correctness: float
    format_score: float
    predicted_answer: str | None


def score_completion(text: str, gold: str, correctness_weight: float, format_weight: float) -> RewardBreakdown:
    predicted = extract_final_answer(text)
    correct = float(predicted is not None and normalize_answer(predicted) == normalize_answer(gold))
    formatted = float("\\boxed{" in text and "}" in text)
    return RewardBreakdown(correctness_weight * correct + format_weight * formatted, correct, formatted, predicted)

