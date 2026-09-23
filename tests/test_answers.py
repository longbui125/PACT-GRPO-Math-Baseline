from pact_grpo.data import extract_final_answer, normalize_answer


def test_extract_numeric_boxed_answer():
    assert extract_final_answer("Therefore \\boxed{42}") == "42"


def test_extract_nested_latex_boxed_answer():
    assert extract_final_answer("Result: \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_normalize_numeric_answer():
    assert normalize_answer("1,200.00") == "1200"

