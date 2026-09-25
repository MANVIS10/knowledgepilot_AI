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

## Task 2: One source of truth for settings + retrieval threshold
Evidence: `config.py` values are duplicated/ignored; answerable scores start at
0.666, off-topic Python question scored 0.627 (> 0.60 threshold).
- Guards and memory import `SIMILARITY_THRESHOLD`, `MAX_QUESTION_LENGTH`,
  `MAX_HISTORY` from `src/config.py`.
- Raise `SIMILARITY_THRESHOLD` to 0.65.
- Split the guard eval: off-topic questions MUST be refused by the guard;
  uncovered-but-in-domain topics overlap answerable scores (0.63-0.75), so they are
  reported, and enforced at application level (LLM refusal).
- Verify: `pytest evals/component/test_retrieval_guard.py`, no wrong refusals.

## Task 3: Follow-up query rewriting
Evidence: follow-up MRR 0.38 / precision 0.35 vs 0.95 / 0.92 standalone.
- When history exists, rewrite the follow-up into a standalone question with one
  small LLM call; use it for retrieval only (the user's original text still goes
  in the prompt). Fall back to the original on any error.
- Verify: follow-up component eval (remove xfail) meets MRR >= 0.70, precision >= 0.50;
  application follow-up tests.

## Task 4 (next): Security evals with DeepTeam
Prompt leakage / injection / jailbreak attacks against the app.
