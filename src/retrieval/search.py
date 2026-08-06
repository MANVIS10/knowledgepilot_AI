import torch
from sentence_transformers import SentenceTransformer

from src.utils.db import get_connection

# Limit PyTorch CPU threads for memory and performance efficiency in container
torch.set_num_threads(1)

# Load embedding model once
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def retrieve_chunks(
    question: str,
    top_k: int = 5
) -> list[str]:
    """
    Retrieve the most relevant chunks from PostgreSQL
    using vector similarity search.
    """

    # Create embedding for the user's question
    with torch.no_grad():
        question_embedding = model.encode(question)


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