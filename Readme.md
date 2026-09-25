# 🧠 KnowledgePilot AI

KnowledgePilot AI is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about **Stanford CS229 (Machine Learning)** using the transcripts of all 20 lectures. It retrieves the most relevant lecture passages with semantic search, has GPT-4o mini answer **only from those passages with numbered citations**, and checks the answer before returning it.

I built it from scratch to understand the whole RAG pipeline, and then **measured it with DeepEval** — component-level and end-to-end — and fixed what the evals found. The results are below.

**Live demo:** https://knowledgepilot-ai.onrender.com · **GitHub:** https://github.com/MANVIS10/knowledgepilot_AI

The free host puts the app to sleep when idle, so the first request after a pause can take a while. You can also run it locally or in Docker (see [Running it](#running-it)).

---

## Features

- Semantic search: `BAAI/bge-small-en-v1.5` embeddings (run with FastEmbed / ONNX, no PyTorch) + PostgreSQL **pgvector**
- Answers grounded in retrieved lecture chunks, with `[1][2]` citations and source cards
- Streaming Gradio chat UI with a **Retrieval-Debug mode** (raw chunks, distances, guardrail status)
- FastAPI `POST /ask` endpoint with Swagger docs
- **Per-user conversation memory** (each browser session / API `session_id` has its own history)
- Three guardrails: input (prompt-injection filter), retrieval (similarity threshold), output (LLM check that the answer is supported by the context)
- Docker image; database hosted on Neon
- DeepEval test suite for components and the full pipeline

---

## Architecture

```
                 User question
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
   Gradio UI (chat)            FastAPI  POST /ask
        └──────────────┬──────────────┘
                       ▼
              KnowledgeBase pipeline
                       │
     1. Input guard ── normalizes text, blocks injection phrasing
                       │
     2. Embed question (bge-small-en-v1.5, FastEmbed/ONNX, 384-dim)
                       │
     3. pgvector cosine search on Neon ── top-5 chunks
                       │
     4. Retrieval guard ── refuse if best similarity < threshold
                       │
     5. Prompt = persona + per-session history + numbered chunks
                       │
     6. GPT-4o mini ── answer with [n] citations (streamed in the UI)
                       │
     7. Output guard ── is the answer supported by the chunks?
                       │
                answer + sources
```

## Tech stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11 |
| UI / API | Gradio, FastAPI |
| LLM | OpenAI GPT-4o mini |
| Embeddings | `BAAI/bge-small-en-v1.5` via FastEmbed (ONNX Runtime) |
| Vector database | PostgreSQL + pgvector (Neon) |
| Evaluation | DeepEval, pytest (DeepTeam installed for the security level) |
| Packaging | Docker |

## Data pipeline (offline)

1. **Parse** – `src/preprocessing/html_parser.py` extracts text from the HTML transcripts.
2. **Chunk** – `chunker.py` splits each lecture into 300-word windows with 50-word overlap → **764 chunks** across 20 lectures.
3. **Embed** – `src/embeddings/embedder.py` creates 384-dim vectors.
4. **Store** – `src/ingestion/store_embeddings.py` loads them into a `documents` table (`VECTOR(384)`). Re-running it replaces a lecture's rows, so it never creates duplicates.

---

## Evaluation results

Every number below comes from the suites in [`evals/`](evals). They use small hand-built sets, so read them as a **baseline, not a benchmark**: 22 answerable questions, 10 unanswerable, 4 follow-ups, 12 attack prompts, 16 legitimate prompts. Relevance labels come from keyword matches in the transcripts: for the 22 answerable questions a chunk counts if it mentions the topic (so those numbers are optimistic); the 4 follow-ups use stricter, concept-specific patterns. Answer quality is judged by `gpt-4o-mini` through DeepEval, and an LLM judge can be lenient.

### Component level

| Component | Metric | Result |
|-----------|--------|--------|
| Retrieval (22 questions, top-5) | Hit rate | **0.95** |
| | MRR | **0.95** |
| | Precision@5 | **0.92** |
| Retrieval, follow-up questions (4, strict labels) | MRR / Precision@5 | **0.25 / 0.25** searched as-is → **0.56 / 0.45** with query rewriting (target 0.70, not yet met) |
| Retrieval guard (threshold 0.60) | Answerable questions wrongly refused | **0 of 22** (lowest answerable score 0.666) |
| | Unanswerable questions let through | **7 of 10** (see below) |
| Input guard | Attack phrasings blocked | **12 of 12** (was 6 of 12 with the original exact-phrase list) |
| | Legitimate questions blocked | **0 of 16** |
| Embedding runner | Vector agreement with the old PyTorch runner | cosine ≥ 0.999998, identical top-5 order |
| | Memory, one model + one query | **217 MB** vs 550 MB before |

### Application level (whole pipeline, DeepEval)

| Check | Result |
|-------|--------|
| Answerable questions (8): faithfulness ≥ 0.7, answer relevancy ≥ 0.7, correctness vs reference ≥ 0.6 | **6 to 8 of 8 passed across runs.** In the last run 2 failed the correctness judge (scores 0.595 and 0.49 against a 0.6 threshold) after passing earlier; the judge is noisy near its threshold |
| Citations: every `[n]` points at a real retrieved source | **8 / 8** (every run) |
| Unanswerable questions (10, off-topic and topics the lectures don't cover) | **10 / 10 refused** (every run) |
| Session isolation: one user's history never reaches another | **passed** |

### What the evals found, and what was fixed

| Problem found | Fix |
|---------------|-----|
| One global memory shared by every user | Memory is now per session |
| The current question appeared twice in the prompt, and blocked questions stayed in memory with no answer | History is read before the question is added; a turn is saved only when an answer is produced |
| Output guard rejected good answers unless the judge replied with exactly `SUPPORTED` | Tolerant verdict parsing, `temperature=0`, and the exact "not enough information" refusal skips the extra LLM call |
| Streamed answers vanished after the output guard failed | The answer stays and a warning is appended |
| Database host and password hard-coded as defaults | All connection settings come from environment variables |
| Input guard missed half of the attack phrasings | Text normalization + regex patterns (12/12 blocked) |
| Follow-up questions were searched without the conversation | One small LLM call rewrites them into standalone queries, used for retrieval only (MRR 0.25 → 0.56) |
| Guard and memory settings duplicated, so editing `config.py` did nothing | Everything reads from `src/config.py` |
| Re-running ingestion duplicated rows | Ingestion replaces a lecture's rows |
| PyTorch made the image large and used ~550 MB RAM | Same model on ONNX: 217 MB |

### Known limitations (found by the evals, not yet fixed)

- **Follow-up questions are only partly fixed.** Query rewriting resolves "it" correctly (MRR 0.25 → 0.56), but retrieval is phrasing-sensitive: "Why does gradient descent need a learning rate?" misses the single chunk that explains the learning rate (Lecture 2, chunk 18), while "What does the learning rate alpha control…" ranks it first. In the end-to-end suite 2 of the 4 follow-ups still fail for this reason and are marked as known gaps.
- **Hybrid search (keyword + vector) was built and measured, and not adopted.** It fixed one standalone question (answerable hit rate 0.95 → 1.00) but lowered follow-up MRR (0.56 → 0.38 with rare-word keyword queries, 0.31 with a naive query), so it stays behind `HYBRID_SEARCH = False`.
- **A similarity threshold can't separate "not covered" from "covered".** Scores for topics the lectures skip (decision trees, BERT, backprop, …) reach 0.75, overlapping answerable questions (from 0.67). The language model's refusal is the real backstop; the retrieval guard only reliably stops clearly off-topic questions.
- **Retrieval is sensitive to phrasing.** "Explain the naive Bayes classifier" missed the main lecture, while a reworded question found it.
- **The input guard is a heuristic filter, not a complete defense.** Security red-teaming (DeepTeam) is the next planned step.
- **The LLM judge is noisy.** The same 8 answerable questions scored 8/8 in one run and 6/8 in another, and the 4 follow-ups swing between 1 and 2 passes, so single-run differences on these small sets are not evidence. Repeating each case and averaging is the next improvement.

---

## API

`POST /ask`

```json
{
  "question": "What is gradient descent?",
  "session_id": "any-string-you-choose"
}
```

`session_id` is optional. Requests with the same id share a conversation; without one, the call is stateless.

Response (abridged):

```json
{
  "answer": "Gradient descent is an iterative optimization algorithm ... [1][2]",
  "sources": [
    {
      "lecture": "Lecture 02",
      "chunk_id": 17,
      "similarity_score": 0.76,
      "text": "..."
    }
  ],
  "retrieved_chunks": ["..."],
  "passed_chunks": ["..."],
  "success": true
}
```

`success` is `false` when a guardrail refused the question or rejected the answer.

---

## Running it

**1. Install**

```bash
git clone https://github.com/MANVIS10/knowledgepilot_AI.git
cd knowledgepilot_AI
pip install -r requirements.txt
```

**2. Configure** – copy `.env.example` to `.env` and fill in your OpenAI key and Postgres details (any Postgres with the pgvector extension, e.g. a Neon project; use `DB_SSLMODE=require` for hosted databases). Missing settings raise a clear error.

**3. Load the data (once)**

```bash
python -m src.embeddings.embedder
python -m src.ingestion.store_embeddings
```

**4. Start it**

```bash
python -m ui.app                  # Gradio UI on http://localhost:7860
uvicorn api.main:app --reload     # API + Swagger on http://localhost:8000/docs
```

**Docker** (runs the Gradio UI):

```bash
docker build -t knowledgepilot .
docker run --env-file .env -p 7860:7860 knowledgepilot
```

## Running the evals

```bash
pip install -r requirements-dev.txt
pytest evals/component      # retrieval, retrieval guard, input guard (no LLM cost)
pytest evals/application    # whole pipeline, calls OpenAI (roughly a few cents to a dollar per run)
```

They need the database loaded and `OPENAI_API_KEY` set. Reports are written to `evals/reports/`.

---

## Project structure

```
├── api/                FastAPI app
├── ui/                 Gradio app
├── src/
│   ├── preprocessing/  HTML parsing, chunking
│   ├── embeddings/     chunk embedding
│   ├── ingestion/      load vectors into Postgres
│   ├── retrieval/      pgvector search, follow-up query rewriting, optional hybrid search
│   ├── generation/     prompt builder, OpenAI generator
│   ├── guardrails/     input, retrieval, output guards
│   ├── memory/         per-session conversation memory
│   ├── pipeline/       KnowledgeBase (ties it all together)
│   └── utils/          database connection
├── evals/              DeepEval component + application suites, datasets
├── docs/plans/         fix plans driven by the eval results
├── data/               raw transcripts, processed chunks
├── Dockerfile
├── requirements.txt          runtime
└── requirements-dev.txt      evals (DeepEval, DeepTeam)
```

## Roadmap

- Cross-encoder re-ranking / retrieving more chunks to fix phrasing-sensitive retrieval
- Revisit the similarity threshold (kept at 0.60: answerable questions score from 0.666, too little margin to raise it) with a larger eval set
- Security evals with DeepTeam (prompt leakage, injection, jailbreaks)
- Hybrid search (BM25 + vectors) and cross-encoder re-ranking
- CI that runs the evals, and a hosted demo

## What I learned

RAG end to end (chunking, embeddings, vector search, prompt design, guardrails), evaluating an LLM app at component and application level with DeepEval, diagnosing failures from data instead of guesses, per-user state in a multi-user app, keeping secrets out of code, and reducing a service's memory footprint without changing its behavior.

---

**Author:** Manvi Soni · [github.com/MANVIS10](https://github.com/MANVIS10)
