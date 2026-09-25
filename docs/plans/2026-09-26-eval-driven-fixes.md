# Eval-driven fixes

Goal: fix the gaps the DeepEval component/application evals found, verifying
each with the eval that exposed it. Work on branch `dev`.

## Task 1: Harden the input guard
Evidence: `evals/component/test_input_guard.py` - 6 of 12 attack phrasings slip through.
- Normalize input before matching: Unicode NFKC, casefold, digit/leetspeak
  substitution, strip punctuation, collapse whitespace.
- Replace exact-phrase list with a few regexes (ignore/disregard ... instructions,
  reveal system prompt, "what were you told", repeat-text-above, pretend-no-rules).
- Add tricky LEGITIMATE questions to `input_allow` so the guard doesn't over-block.
- Verify: `pytest evals/component/test_input_guard.py` all green.
- Limit to state: a heuristic filter, not a complete defense (DeepTeam will probe it).

## Task 2: One source of truth for settings (threshold kept at 0.60)
Evidence: `config.py` values were duplicated/ignored by the guards and memory.
- Guards and memory import `SIMILARITY_THRESHOLD`, `MAX_QUESTION_LENGTH`,
  `MAX_HISTORY` from `src/config.py`.
- DECISION CHANGE (found in review): the plan first said raise the threshold to
  0.65. Not done: the lowest answerable score is 0.666 (margin 0.016 on 22 questions),
  so 0.65 risks refusing real questions, and it would only additionally stop 1 of 7
  unanswerable questions that the language model already refuses (10/10 at
  application level). Keep 0.60; revisit with a larger eval set.
- Guard eval now asserts: no wrong refusals + >= 0.05 margin above the threshold.
  Uncovered-but-in-domain topics overlap answerable scores (0.63-0.75), so they are
  reported, and enforced at application level (LLM refusal).
- Verify: `pytest evals/component`, no wrong refusals.

## Task 3: Follow-up query rewriting
Evidence: follow-up MRR 0.38 / precision 0.35 vs 0.95 / 0.92 standalone.
- When history exists, rewrite the follow-up into a standalone question with one
  small LLM call; use it for retrieval only (the user's original text still goes
  in the prompt). Fall back to the original on any error.
- Verify: follow-up component eval (remove xfail) meets MRR >= 0.70, precision >= 0.50;
  application follow-up tests.

## Task 3 result
Follow-up component eval (4 cases): MRR 0.38 -> 0.81, precision@5 0.35 -> 0.80.
Application-level: 2 of 4 follow-ups still fail. Cause is NOT the rewrite (it resolves
"it" correctly) but retrieval phrasing sensitivity: the one chunk that explains the
learning rate (Lecture 2, chunk 18) is missed by "Why does gradient descent need a
learning rate?" but found first by "What does the learning rate alpha control...".
The component labels were also too loose (any "gradient descent" chunk counted as
relevant), so 0.81 overstates the gain.

## Task 4: Tighten follow-up relevance labels
Use strict per-case patterns (the specific concept, not just the topic) and verify each
matches the intended chunks. Re-measure baseline vs rewritten.

## Task 5: Hybrid search (keyword + vector)
Evidence: exact terms ("learning rate") are missed by pure embedding search.
- Postgres full-text (tsvector + GIN) alongside pgvector; fuse rankings with
  Reciprocal Rank Fusion. Keep each chunk's vector `distance` so the retrieval guard
  still works (guard uses the BEST distance, not the first chunk's).
- `HYBRID_SEARCH` flag in config; the eval measures vector-only vs hybrid on the same
  cases and we keep hybrid only if the numbers justify it.

## Task 6 (later): Security evals with DeepTeam
Prompt leakage / injection / jailbreak attacks against the app.
