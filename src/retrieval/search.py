import numpy as np
from fastembed import TextEmbedding

from src.utils.db import get_connection

# Load embedding model once (same model as before, run with ONNX instead of PyTorch).
# One CPU thread keeps memory and CPU use low in a small container.
model = TextEmbedding("BAAI/bge-small-en-v1.5", threads=1)


def retrieve_chunks(
    question: str,
    top_k: int = 5
) -> list[str]:
    """
    Retrieve the most relevant chunks from PostgreSQL
    using vector similarity search.
    """

    # Create embedding for the user's question
    question_embedding = np.array(next(iter(model.embed([question]))))


    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
      SELECT
       lecture_name,
       chunk_id,
       chunk_text,
       embedding <=> %s::vector AS distance
       FROM documents
       ORDER BY distance
       LIMIT %s;
        """,
        (
            question_embedding.tolist(),
            top_k
        )
    )

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