import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def load_embeddings(input_file: Path) -> np.ndarray:
    embeddings = np.load(input_file)
    return embeddings
   



def load_chunks(input_file: Path) -> list[str]:
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = []

    for chunk in data:
        texts.append(chunk["text"])

    return texts


def embed_query(
    query: str,
    model: SentenceTransformer
) -> np.ndarray:

    query_embedding = model.encode(query)

    return query_embedding


def cosine_similarity(
    query_embedding: np.ndarray,
    embeddings: np.ndarray
    ) -> np.ndarray:

     dot_products = np.dot(
        embeddings,
        query_embedding
    )

     query_norm = np.linalg.norm(
        query_embedding
    )

     embedding_norms = np.linalg.norm(
        embeddings,
        axis=1
    )

     similarities = (
        dot_products /
        (query_norm * embedding_norms)
    )

     return similarities



def retrieve_top_k(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    chunks: list[str],
    k: int = 3
) -> list[dict]:

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )

    top_k_indices = np.argsort(
        similarities
    )[::-1][:k]

    results = []

    for index in top_k_indices:
        results.append(
            {
                "chunk": chunks[index],
                "score": float(similarities[index])
            }
        )

    return results

def main():

    embeddings_file = Path(
        "data/embeddings/lecture01_embeddings.npy"
    )

    chunks_file = Path(
        "data/processed/lecture01_chunks.json"
    )

    embeddings = load_embeddings(
        embeddings_file
    )

    chunks = load_chunks(
        chunks_file
    )

    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5"
    )

    query = input("Enter your question: ")

    query_embedding = embed_query(
        query,
        model
    )

    results = retrieve_top_k(
        query_embedding,
        embeddings,
        chunks
    )

    print("\nTop Retrieved Chunks:\n")

    for i, result in enumerate(results, start=1):
        print(f"Chunk {i}")
        print(f"Similarity Score: {result['score']:.4f}")
        print("-" * 50)
        print(result["chunk"])
        print()


if __name__ == "__main__":
    main()