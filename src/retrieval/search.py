import re
from collections import Counter
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from src.config import (
    HYBRID_CANDIDATES,
    HYBRID_SEARCH,
    KEYWORD_MAX_DOC_FREQ,
    KEYWORD_MAX_TERMS,
    RRF_K,
)
from src.utils.db import get_connection

# Load embedding model once (same model as before, run with ONNX instead of PyTorch).
# One CPU thread keeps memory and CPU use low in a small container.
model = TextEmbedding("BAAI/bge-small-en-v1.5", threads=1)


@lru_cache(maxsize=1)
def word_document_frequencies() -> tuple[Counter, int]:
    """How many chunks each word appears in (loaded once, ~0.5 MB of text)."""

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT chunk_text FROM documents")
    texts = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()

    frequencies = Counter()
    for text in texts:
        frequencies.update(set(re.findall(r"[a-z0-9]+", text.lower())))

    return frequencies, len(texts)


def keyword_query(question: str) -> str:
    """
    Turn a question into a Postgres full-text query where ANY word may match
    ("rate | descent"). Only the RARE words are kept: words that appear in
    most chunks ("does", "need", "learning" in an ML course) add noise, and
    Postgres ranking has no notion of word rarity. ANDing every word would
    match nothing for a natural-language question.
    """

    frequencies, total = word_document_frequencies()

    words = dict.fromkeys(re.findall(r"[a-z0-9]+", question.lower()))
    rare = [
        w for w in words
        if len(w) > 2 and 0 < frequencies[w] / total <= KEYWORD_MAX_DOC_FREQ
    ]
    rare.sort(key=lambda w: frequencies[w])

    return " | ".join(rare[:KEYWORD_MAX_TERMS])


# Vector search and keyword search each rank their own top candidates; a chunk's
# final score is the sum of 1 / (RRF_K + rank) over the lists it appears in
# (Reciprocal Rank Fusion). Every returned chunk keeps its real vector distance,
# which the retrieval guard needs.
HYBRID_SQL = """
    WITH q AS (
        SELECT %(vec)s::vector AS v, to_tsquery('english', %(kw)s) AS t
    ),
    vec AS (
        SELECT d.id, ROW_NUMBER() OVER (ORDER BY d.embedding <=> q.v) AS r
        FROM documents d, q
        ORDER BY d.embedding <=> q.v
        LIMIT %(cand)s
    ),
    kw AS (
        SELECT d.id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(d.ts, q.t, 1) DESC) AS r
        FROM documents d, q
        WHERE d.ts @@ q.t
        ORDER BY ts_rank_cd(d.ts, q.t, 1) DESC
        LIMIT %(cand)s
    )
    SELECT d.lecture_name, d.chunk_id, d.chunk_text, d.embedding <=> q.v AS distance
    FROM documents d
    CROSS JOIN q
    LEFT JOIN vec ON vec.id = d.id
    LEFT JOIN kw ON kw.id = d.id
    WHERE vec.id IS NOT NULL OR kw.id IS NOT NULL
    ORDER BY COALESCE(1.0 / (%(rrf)s + vec.r), 0) + COALESCE(1.0 / (%(rrf)s + kw.r), 0) DESC
    LIMIT %(k)s;
"""

VECTOR_SQL = """
    SELECT lecture_name, chunk_id, chunk_text, embedding <=> %(vec)s::vector AS distance
    FROM documents
    ORDER BY distance
    LIMIT %(k)s;
"""


def retrieve_chunks(
    question: str,
    top_k: int = 5,
    hybrid: bool | None = None
) -> list[dict]:
    """
    Retrieve the most relevant chunks from PostgreSQL.

    hybrid=True  -> vector search + keyword search merged with RRF
    hybrid=False -> vector search only
    hybrid=None  -> use the HYBRID_SEARCH setting in src/config.py
    """

    if hybrid is None:
        hybrid = HYBRID_SEARCH

    # Create embedding for the user's question
    question_embedding = np.array(next(iter(model.embed([question]))))

    params = {"vec": question_embedding.tolist(), "k": top_k}

    if hybrid:
        params.update(
            kw=keyword_query(question),
            cand=HYBRID_CANDIDATES,
            rrf=RRF_K,
        )
        sql = HYBRID_SQL
    else:
        sql = VECTOR_SQL

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(sql, params)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "lecture": row[0],
            "chunk_id": row[1],
            "text": row[2],
            "distance": float(row[3]),
        }
        for row in rows
    ]
