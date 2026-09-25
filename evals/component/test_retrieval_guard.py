"""
COMPONENT EVAL: retrieval guard (similarity threshold).

Question: does one cut-off separate questions the lectures can answer from
questions they can't? Measures wrong refusals and wrong acceptances.
"""
from evals.common import load_cases, save_report
from src.guardrails.retrieval_guard import SIMILARITY_THRESHOLD, validate_retrieval
from src.retrieval.search import retrieve_chunks

CASES = load_cases()


def guard_passes(question):
    chunks = retrieve_chunks(question, 5)
    allowed, _ = validate_retrieval(question, chunks)
    return allowed, round(1 - chunks[0]["distance"], 3)


def test_threshold_separates_answerable_from_unanswerable():
    answerable = [(c["question"],) + guard_passes(c["question"]) for c in CASES["answerable"]]
    unanswerable = [(c["question"], c["kind"]) + guard_passes(c["question"]) for c in CASES["unanswerable"]]

    wrongly_refused = [(q, s) for q, ok, s in answerable if not ok]
    wrongly_accepted = [(q, kind, s) for q, kind, ok, s in unanswerable if ok]
    by_kind = {}
    for q, kind, ok, s in unanswerable:
        d = by_kind.setdefault(kind, [0, 0]); d[0] += ok; d[1] += 1

    lowest_ok = min(s for _, _, s in answerable)
    highest_bad = max(s for _, _, _, s in unanswerable)
    save_report("retrieval_guard", {
        "threshold": SIMILARITY_THRESHOLD,
        "wrongly_refused": wrongly_refused, "wrongly_accepted": wrongly_accepted,
        "lowest_answerable_score": lowest_ok, "highest_unanswerable_score": highest_bad,
    })
    print(f"\nthreshold={SIMILARITY_THRESHOLD}  "
          f"answerable scores start at {lowest_ok}, unanswerable go up to {highest_bad}")
    print("wrongly refused (should answer):", wrongly_refused)
    for kind, (acc, total) in by_kind.items():
        print(f"  {kind}: {acc}/{total} wrongly accepted")
    print("wrongly accepted:", wrongly_accepted)

    # What a similarity threshold can guarantee: never refuse a question the
    # lectures answer, and keep a safety margin above the cut-off.
    assert not wrongly_refused, "guard refuses questions the lectures can answer"
    assert lowest_ok >= SIMILARITY_THRESHOLD + 0.05, (
        f"lowest answerable score {lowest_ok} is too close to threshold {SIMILARITY_THRESHOLD}"
    )

    # What it cannot guarantee: scores for topics the lectures skip (up to ~0.75)
    # overlap answerable ones (from ~0.67), so those are only REPORTED here.
    # The language model's refusal is the backstop and is enforced at application
    # level (evals/application: test_unanswerable_question_is_refused).
