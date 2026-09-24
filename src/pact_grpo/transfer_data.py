"""Disjoint equation families for retention versus adaptation experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .radical_equations import RadicalCase, _case, candidate_roots, exact_roots


@dataclass(frozen=True)
class TransferSplits:
    easy_train: list[RadicalCase]
    easy_validation: list[RadicalCase]
    easy_test: list[RadicalCase]
    hard_train: list[RadicalCase]
    hard_validation: list[RadicalCase]
    hard_test: list[RadicalCase]


EASY_COUNTS = {
    "train": (180, 500, 120),
    "validation": (80, 80, 40),
    "test": (200, 100, 50),
}
HARD_COUNTS = {"train": 500, "validation": 100, "test": 200}


def _pool(k_values: range, m_values: tuple[int, ...], b_values: range,
          a_values: range, max_root: int) -> dict[int, list[tuple]]:
    pools: dict[int, list[tuple]] = {0: [], 1: [], 2: []}
    for k in k_values:
        for m in m_values:
            for b in b_values:
                for a in a_values:
                    candidates = candidate_roots(k, a, m, b)
                    if candidates is None or not candidates or any(abs(x) > max_root for x in candidates):
                        continue
                    roots = exact_roots(k, a, m, b)
                    assert roots is not None
                    pools[len(roots)].append((k, a, m, b, roots))
    return pools


def make_transfer_splits(seed: int = 42) -> TransferSplits:
    """A uses slopes ±1/±2; B uses disjoint, steeper slopes ±3/±4.

    B contains exactly two squared-equation candidates, one extraneous. The
    fixed family definitions and counts do not depend on model outcomes.
    """
    easy = _pool(range(1, 11), (-2, -1, 1, 2), range(-20, 21),
                 range(0, 201), max_root=30)
    hard = _pool(range(4, 13), (-4, -3, 3, 4), range(-30, 31),
                 range(0, 401), max_root=50)
    hard_mixed = [item for item in hard[1]
                  if len(candidate_roots(item[0], item[1], item[2], item[3]) or ()) == 2]
    rng = random.Random(seed)
    for items in easy.values():
        rng.shuffle(items)
    rng.shuffle(hard_mixed)
    easy_splits = {}
    for split, counts in EASY_COUNTS.items():
        selected = []
        for root_count, count in enumerate(counts):
            if len(easy[root_count]) < count:
                raise ValueError(f"Easy family lacks {root_count}-root cases for {split}")
            selected.extend(_case(*easy[root_count].pop(), f"easy-{split}")
                            for _ in range(count))
        rng.shuffle(selected)
        easy_splits[split] = selected
    hard_splits = {}
    for split, count in HARD_COUNTS.items():
        if len(hard_mixed) < count:
            raise ValueError(f"Hard family lacks mixed-root cases for {split}")
        selected = [_case(*hard_mixed.pop(), f"hard-{split}") for _ in range(count)]
        rng.shuffle(selected)
        hard_splits[split] = selected
    return TransferSplits(
        easy_splits["train"], easy_splits["validation"], easy_splits["test"],
        hard_splits["train"], hard_splits["validation"], hard_splits["test"],
    )
