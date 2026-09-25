"""COMPONENT EVAL: input guard (blocked phrases and length limits)."""
import pytest

from evals.common import load_cases
from src.guardrails.input_guard import validate_question

CASES = load_cases()


@pytest.mark.parametrize("question", CASES["input_allow"])
def test_legitimate_questions_are_allowed(question):
    allowed, _ = validate_question(question)
    assert allowed


@pytest.mark.parametrize("question", CASES["input_block"])
def test_attacks_are_blocked(question):
    allowed, _ = validate_question(question)
    assert not allowed, f"not blocked: {question!r}"


def test_length_limits():
    assert not validate_question("")[0]
    assert not validate_question("   ")[0]
    assert validate_question("a" * 500)[0]
    assert not validate_question("a" * 501)[0]
