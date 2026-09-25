import json
from pathlib import Path

import numpy as np

from src.utils.db import get_connection


def process_lecture(
    chunk_file: Path,
    embedding_file: Path,
):

    lecture_name = chunk_file.stem.replace("_chunks", "")

    with open(chunk_file, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    embeddings = np.load(embedding_file)

    conn = get_connection()
    cur = conn.cursor()

    # Create pgvector extension
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
    """)

    # Create documents table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            lecture_name TEXT,
            chunk_id INTEGER,
            chunk_text TEXT,
            embedding VECTOR(384)
        );
    """)

    conn.commit()

    # Replace this lecture's rows so re-running never creates duplicates
    cur.execute(
        "DELETE FROM documents WHERE lecture_name = %s",
        (lecture_name,),
    )

    # Insert embeddings
    for chunk, embedding in zip(chunks, embeddings):

        cur.execute(
            """
            INSERT INTO documents
            (lecture_name, chunk_id, chunk_text, embedding)
            VALUES (%s, %s, %s, %s)
            """,
            (
                lecture_name,
                chunk["chunk_id"],
                chunk["text"],
                embedding.tolist(),
            ),
        )

    conn.commit()

    cur.close()
    conn.close()

    print(f"Stored {lecture_name}")


def main():

    processed_folder = Path("data/processed")
    embedding_folder = Path("data/embeddings")

    for chunk_file in sorted(processed_folder.glob("*_chunks.json")):

        embedding_file = (
            embedding_folder
            / f"{chunk_file.stem.replace('_chunks','')}_embeddings.npy"
        )

        process_lecture(
            chunk_file,
            embedding_file,
        )


if __name__ == "__main__":
    main()