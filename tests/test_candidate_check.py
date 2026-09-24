import json
from pathlib import Path

from pact_grpo.data import ANSWER_PREFIX, format_prompt
from pact_grpo.radical_equations import candidate_roots, make_radical_splits, parse_solution_set
from pact_grpo.radical_rewards import candidate_overlap, verified_reward


class DummyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "chat:"


def test_prompt_and_first_answer_box():
    assert format_prompt("Solve x=1", DummyTokenizer()) == "chat:" + ANSWER_PREFIX
    assert parse_solution_set(r"\boxed{3} then \boxed{4}") == frozenset({3})


def test_candidate_overlap_rewards_partial_but_exact_dominates():
    case = next(case for case in make_radical_splits()[0] if len(case.roots) == 2)
    root = min(case.roots)
    assert candidate_overlap(case.roots, case) == 1.0
    assert candidate_overlap(frozenset({root}), case) == 0.5
    assert (
        verified_reward(f"\\boxed{{{case.example.answer}}}", case)
        > verified_reward(f"\\boxed{{{root}}}", case)
        > verified_reward(r"\boxed{none}", case)
    )
    invalid = max(candidate_roots(case.k, case.a, case.m, case.b) or ()) + 1
    assert verified_reward(f"\\boxed{{{invalid}}}", case) == 0
    assert verified_reward(r"\boxed{1,2,3}", case) == 0


def test_math_sft_file_is_training_only():
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "math_algebra_sft.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((path.parent / "math_algebra_sft_manifest.json").read_text())
    assert len(rows) == manifest["selected"] == 600
    assert manifest["split"] == "train"
    assert len({row["source_index"] for row in rows}) == len(rows)
    assert all(row["prompt_tokens"] + row["response_tokens"] <= 256 for row in rows)
