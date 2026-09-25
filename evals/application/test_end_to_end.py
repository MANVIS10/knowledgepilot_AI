"""
APPLICATION EVAL: the whole pipeline through KnowledgeBase.ask()
(input guard -> retrieval -> retrieval guard -> generation -> output guard).

Uses DeepEval with gpt-4o-mini as the judge. Cases:
  - answerable questions: faithfulness, answer relevancy, correctness, citations
  - unanswerable questions: the app must refuse instead of making things up
  - follow-ups: history must carry the topic across turns
  - session isolation: one user's history must not leak to another
"""
import json
import re
import uuid
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from evals.common import load_cases
from src.pipeline.knowledge_base import KnowledgeBase

JUDGE = "gpt-4o-mini"
APP = json.loads((Path(__file__).parents[1] / "datasets" / "application_cases.json").read_text(encoding="utf-8"))
COMPONENT = load_cases()

REFUSAL_MARKERS = (
    "don't have enough information",
    "couldn't find relevant information",
    "couldn't verify",
    "isn't supported",
)


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


def ask(kb, question, session=None):
    return kb.ask(question, session_id=session or str(uuid.uuid4()))


def is_refusal(result) -> bool:
    text = result["answer"].lower()
    return (not result["success"]) or any(m in text for m in REFUSAL_MARKERS)


def test_case_from(question, result, expected=None):
    return LLMTestCase(
        input=question,
        actual_output=result["answer"],
        expected_output=expected,
        retrieval_context=[c["text"] for c in result["retrieved_chunks"]],
    )


# ---------------------------------------------------------------- answerable
@pytest.mark.parametrize("case", APP["answerable"], ids=lambda c: c["question"])
def test_answerable_question(kb, case):
    result = ask(kb, case["question"])
    assert not is_refusal(result), f"refused an answerable question: {result['answer']}"

    correctness = GEval(
        name="Correctness",
        criteria="The actual output conveys the same key facts as the expected output "
                 "and does not contradict it. Extra correct detail is fine.",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        threshold=0.6, model=JUDGE,
    )
    assert_test(
        test_case_from(case["question"], result, case["expected_output"]),
        [
            FaithfulnessMetric(threshold=0.7, model=JUDGE),
            AnswerRelevancyMetric(threshold=0.7, model=JUDGE),
            correctness,
        ],
    )


@pytest.mark.parametrize("case", APP["answerable"], ids=lambda c: c["question"])
def test_citations_point_at_real_sources(kb, case):
    result = ask(kb, case["question"])
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", result["answer"])}
    assert cited, "answer has no [n] citations"
    assert all(1 <= n <= len(result["sources"]) for n in cited), f"bad citation numbers {cited}"


# -------------------------------------------------------------- unanswerable
@pytest.mark.parametrize("case", COMPONENT["unanswerable"], ids=lambda c: c["question"])
def test_unanswerable_question_is_refused(kb, case):
    result = ask(kb, case["question"])
    assert is_refusal(result), f"answered a question the lectures don't cover: {result['answer'][:200]}"


# ----------------------------------------------------------------- follow-ups
@pytest.mark.xfail(reason="Known gap: follow-ups are retrieved without history "
                          "(no query rewriting yet)", strict=False)
@pytest.mark.parametrize("case", COMPONENT["followups"], ids=lambda c: c["question"])
def test_followup_uses_conversation_history(kb, case):
    session = str(uuid.uuid4())
    for earlier in case["history"]:
        ask(kb, earlier, session)
    result = ask(kb, case["question"], session)
    assert not is_refusal(result), f"lost the topic: {result['answer'][:200]}"
    assert_test(
        test_case_from(case["question"], result),
        [GEval(
            name="Follow-up handling",
            criteria=f"The user first asked: {case['history'][0]!r}. The actual output answers the "
                     "follow-up about that same topic instead of saying it lacks information.",
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            threshold=0.6, model=JUDGE,
        )],
    )


# ---------------------------------------------------------------- isolation
def test_sessions_do_not_share_history(kb):
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    ask(kb, "What is gradient descent?", a)
    assert kb.get_memory(b).get_messages() == []
    assert len(kb.get_memory(a).get_messages()) == 2
