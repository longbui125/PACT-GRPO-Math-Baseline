"""Benchmark for checking candidate roots after squaring radical equations."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

from .data import MathExample


@dataclass(frozen=True)
class RadicalCase:
    k: int
    a: int
    m: int
    b: int
    roots: frozenset[int]
    example: MathExample


def candidate_roots(k: int, a: int, m: int, b: int) -> tuple[int, ...] | None:
    """Roots after squaring; None if any root is non-integer."""
    quadratic = m * m
    linear = 2 * m * b - k
    constant = b * b - a
    discriminant = linear * linear - 4 * quadratic * constant
    if discriminant < 0:
        return ()
    square_root = math.isqrt(discriminant)
    if square_root * square_root != discriminant:
        return None
    numerators = (-linear + square_root, -linear - square_root)
    denominator = 2 * quadratic
    if any(numerator % denominator for numerator in numerators):
        return None
    return tuple(sorted({numerator // denominator for numerator in numerators}))


def root_satisfies_original(k: int, a: int, m: int, b: int, root: int) -> bool:
    radicand = k * root + a
    right = m * root + b
    return radicand >= 0 and right >= 0 and right * right == radicand


def exact_roots(k: int, a: int, m: int, b: int) -> frozenset[int] | None:
    candidates = candidate_roots(k, a, m, b)
    if candidates is None:
        return None
    return frozenset(
        root for root in candidates if root_satisfies_original(k, a, m, b, root)
    )


def _equation(k: int, a: int, m: int, b: int) -> str:
    left = f"{k}x" if k != 1 else "x"
    if a:
        left += f" + {a}"
    right = "x" if m == 1 else "-x" if m == -1 else f"{m}x"
    if b:
        right += f" + {b}" if b > 0 else f" - {-b}"
    return f"sqrt({left}) = {right}"


def _answer(roots: frozenset[int]) -> str:
    return "none" if not roots else ",".join(str(value) for value in sorted(roots))


def _case(k: int, a: int, m: int, b: int,
          roots: frozenset[int], split: str) -> RadicalCase:
    uid = f"radical-{split}-{k}-{a}-{m}-{b}"
    candidates = candidate_roots(k, a, m, b)
    assert candidates is not None and candidates
    candidate_text = ", ".join(str(value) for value in candidates)
    question = (
        f"Squaring both sides of {_equation(k, a, m, b)} gives candidate roots "
        f"{candidate_text}. Substitute each candidate into the ORIGINAL equation "
        "and return only the valid roots; write none if every candidate is extraneous."
    )
    example = MathExample(uid, question, _answer(roots), "radical_equation")
    return RadicalCase(k, a, m, b, roots, example)


def make_radical_splits(seed: int = 42, train_count: int = 500,
                        validation_count: int = 100, test_count: int = 200
                        ) -> tuple[list[RadicalCase], list[RadicalCase], list[RadicalCase]]:
    """Disjoint parameter splits stratified by zero, one, or two real roots."""
    pools: dict[int, list[tuple[int, int, int, int, frozenset[int]]]] = {
        0: [], 1: [], 2: [],
    }
    for k in range(1, 11):
        for m in (-3, -2, -1, 1, 2, 3):
            for b in range(-12, 13):
                for a in range(0, 161):
                    candidates = candidate_roots(k, a, m, b)
                    if candidates is None or not candidates or any(abs(x) > 20 for x in candidates):
                        continue
                    roots = exact_roots(k, a, m, b)
                    assert roots is not None
                    pools[len(roots)].append((k, a, m, b, roots))
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    results = []
    for split, count in (("train", train_count), ("validation", validation_count),
                         ("test", test_count)):
        counts = [round(count * 0.2), round(count * 0.6)]
        counts.append(count - sum(counts))
        selected = []
        for root_count, amount in enumerate(counts):
            if len(pools[root_count]) < amount:
                raise ValueError(f"Insufficient {root_count}-root cases")
            selected.extend(_case(*pools[root_count].pop(), split) for _ in range(amount))
        rng.shuffle(selected)
        results.append(selected)
    return tuple(results)


def parse_solution_set(completion: str) -> frozenset[int] | None:
    matches = re.findall(r"\\boxed\{([^{}]*)\}", completion)
    if not matches:
        return None
    content = matches[0].strip().lower()
    if content in {"none", "no solution", "no solutions", "empty", "\\varnothing", "\\emptyset", "∅"}:
        return frozenset()
    content = content.replace("\\{", "").replace("\\}", "").strip("{} ")
    content = re.sub(r"\bx\s*=\s*", "", content)
    content = content.replace("\\text{and}", ",").replace(" and ", ",").replace(";", ",")
    if not re.fullmatch(r"-?\d+(?:\s*,\s*-?\d+)*", content):
        return None
    values = [int(part.strip()) for part in content.split(",")]
    return frozenset(values) if len(values) == len(set(values)) else None


def sound_solution(completion: str, case: RadicalCase) -> bool:
    roots = parse_solution_set(completion)
    if roots is None:
        return False
    if not roots:
        return not case.roots
    return all(root_satisfies_original(case.k, case.a, case.m, case.b, root)
               for root in roots)


def exact_solution(completion: str, case: RadicalCase) -> bool:
    roots = parse_solution_set(completion)
    return roots is not None and roots == case.roots
