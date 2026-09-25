from pathlib import Path
from src.generation.generator import Generator
from fastembed import TextEmbedding

from src.retrieval.retriever import (
    load_embeddings,
    load_chunks,
    embed_query,
    retrieve_top_k,
)

from src.generation.prompt_builder import build_prompt
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

    model = TextEmbedding(
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

    retrieved_chunks = [
        result["chunk"]
        for result in results
    ]

    print("\nTop Retrieved Chunks:\n")

    for i, result in enumerate(results, start=1):
        print(f"Chunk {i} (Score: {result['score']:.4f})")
        print(result["chunk"])
        print("-" * 80)

    generator = Generator()

    answer = generator.generate(
        query,
        retrieved_chunks
    )

    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()