"""
COMPONENT EVAL: retrieval (embedding + pgvector search).

No LLM calls, so it costs nothing. A retrieved chunk is "relevant" when it
contains the topic's keywords (see datasets/component_cases.json).
"""
import pytest

from evals.common import is_relevant, load_cases, save_report
from src.retrieval.search import retrieve_chunks

K = 5
CASES = load_cases()


def score(cases):
    rows = []
    for case in cases:
        chunks = retrieve_chunks(case["question"], K)
        flags = [is_relevant(c["text"], case["relevant_regex"]) for c in chunks]
        first = next((i for i, f in enumerate(flags, start=1) if f), None)
        rows.append({
            "question": case["question"],
            "hit": first is not None,
            "reciprocal_rank": 1 / first if first else 0.0,
            "precision": sum(flags) / K,
            "top_similarity": round(1 - chunks[0]["distance"], 3),
            "retrieved": [f'{c["lecture"][-2:]}:{c["chunk_id"]}' for c in chunks],
        })
    n = len(rows)
    return {
        "hit_rate": sum(r["hit"] for r in rows) / n,
        "mrr": sum(r["reciprocal_rank"] for r in rows) / n,
        "precision_at_k": sum(r["precision"] for r in rows) / n,
        "rows": rows,
    }


@pytest.fixture(scope="module")
def answerable():
    result = score(CASES["answerable"])
    save_report("retrieval_answerable", result)
    print(f"\nhit@{K}={result['hit_rate']:.2f}  MRR={result['mrr']:.2f}  "
          f"precision@{K}={result['precision_at_k']:.2f}")
    for r in result["rows"]:
        if not r["hit"] or r["precision"] < 0.4:
            print("  WEAK:", r["question"], r["retrieved"])
    return result


def test_hit_rate(answerable):
    assert answerable["hit_rate"] >= 0.90


def test_mrr(answerable):
    assert answerable["mrr"] >= 0.80


def test_precision(answerable):
    assert answerable["precision_at_k"] >= 0.60


@pytest.mark.xfail(reason="Known gap: follow-ups are searched without the "
                          "conversation history (no query rewriting yet)", strict=False)
def test_followup_hit_rate():
    result = score(CASES["followups"])
    save_report("retrieval_followups", result)
    print(f"\nfollow-up hit@{K}={result['hit_rate']:.2f}  MRR={result['mrr']:.2f}")
    # hit-rate alone is too lenient here: check the ranking quality too
    assert result["mrr"] >= 0.70
    assert result["precision_at_k"] >= 0.50
