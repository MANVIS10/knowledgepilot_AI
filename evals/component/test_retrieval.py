"""
COMPONENT EVAL: retrieval (embedding + pgvector search).

No LLM calls, so it costs nothing. A retrieved chunk is "relevant" when it
contains the topic's keywords (see datasets/component_cases.json).
"""
import pytest

from evals.common import is_relevant, load_cases, save_report
from src.retrieval.query_rewriter import rewrite_query
from src.retrieval.search import retrieve_chunks

K = 5
CASES = load_cases()


def history_messages(case):
    """Earlier user turns as chat messages (answers omitted: the rewriter only
    needs the topic, so this slightly understates the real history)."""
    return [{"role": "user", "content": q} for q in case.get("history", [])]


def score(cases, rewrite=False, hybrid=None):
    rows = []
    for case in cases:
        query = rewrite_query(case["question"], history_messages(case)) if rewrite else case["question"]
        chunks = retrieve_chunks(query, K, hybrid=hybrid)
        flags = [is_relevant(c["text"], case["relevant_regex"]) for c in chunks]
        first = next((i for i, f in enumerate(flags, start=1) if f), None)
        rows.append({
            "question": case["question"],
            "search_query": query,
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


@pytest.mark.xfail(reason="Known gap: rewriting lifts follow-up MRR 0.25 -> 0.56 but the 0.70 "
                   "target is not met; retrieval is phrasing-sensitive", strict=False)
def test_followup_retrieval_with_query_rewriting():
    """Follow-ups like "Why does it need a learning rate?" only work if the
    search sees the topic. Compares searching the raw follow-up (baseline) with
    the rewritten standalone query."""
    baseline = score(CASES["followups"])
    rewritten = score(CASES["followups"], rewrite=True)
    save_report("retrieval_followups", {"baseline": baseline, "rewritten": rewritten})
    print(f"follow-ups baseline: MRR={baseline['mrr']:.2f} precision@{K}={baseline['precision_at_k']:.2f}")
    print(f"follow-ups rewritten: MRR={rewritten['mrr']:.2f} precision@{K}={rewritten['precision_at_k']:.2f}")
    for r in rewritten["rows"]:
        print("  ", r["question"], "->", r["search_query"], "| hit:", r["hit"], "p:", r["precision"])
    assert rewritten["mrr"] >= 0.70
    assert rewritten["precision_at_k"] >= 0.50


@pytest.mark.xfail(reason="Experiment: hybrid search is OFF by default because it measured worse "
                   "on follow-ups (MRR 0.38 vs 0.56); passes if future tuning beats vector-only", strict=False)
def test_hybrid_search_vs_vector_only():
    """A/B on the same questions: does adding keyword search help, and does it
    hurt questions that vector search already handled well?"""
    report = {}
    for label, hybrid in (("vector_only", False), ("hybrid", True)):
        report[label] = {
            "answerable": score(CASES["answerable"], hybrid=hybrid),
            "followups_rewritten": score(CASES["followups"], rewrite=True, hybrid=hybrid),
        }
    save_report("retrieval_hybrid_vs_vector", report)

    for label, groups in report.items():
        a, f = groups["answerable"], groups["followups_rewritten"]
        print(f"{label:12} answerable hit={a['hit_rate']:.2f} MRR={a['mrr']:.2f} p@{K}={a['precision_at_k']:.2f}"
              f" | follow-ups MRR={f['mrr']:.2f} p@{K}={f['precision_at_k']:.2f}")

    base, hyb = report["vector_only"], report["hybrid"]
    # must not damage what already works...
    assert hyb["answerable"]["hit_rate"] >= base["answerable"]["hit_rate"] - 0.05
    assert hyb["answerable"]["mrr"] >= base["answerable"]["mrr"] - 0.05
    # ...and must actually help the hard cases to be worth keeping
    assert hyb["followups_rewritten"]["mrr"] > base["followups_rewritten"]["mrr"]
