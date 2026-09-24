from pact_grpo.radical_equations import candidate_roots, exact_roots
from pact_grpo.transfer_data import make_transfer_splits


def test_transfer_families_are_disjoint_and_labeled():
    splits = make_transfer_splits(42)
    expected = {
        "easy_train": 800, "easy_validation": 200, "easy_test": 350,
        "hard_train": 500, "hard_validation": 100, "hard_test": 200,
    }
    seen = set()
    for name, count in expected.items():
        cases = getattr(splits, name)
        assert len(cases) == count
        for case in cases:
            key = (case.k, case.a, case.m, case.b)
            assert key not in seen
            seen.add(key)
            assert case.roots == exact_roots(*key)
            assert set(case.roots).issubset(candidate_roots(*key))
            if name.startswith("easy"):
                assert abs(case.m) in {1, 2}
            else:
                assert abs(case.m) in {3, 4}
                assert len(case.roots) == 1
                assert len(candidate_roots(*key)) == 2

    for name, counts in {
        "easy_train": (180, 500, 120),
        "easy_validation": (80, 80, 40),
        "easy_test": (200, 100, 50),
    }.items():
        cases = getattr(splits, name)
        assert tuple(sum(len(case.roots) == n for case in cases) for n in range(3)) == counts
