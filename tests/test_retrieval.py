from src.retrieval.search import retrieve_chunks


def main():

    question = "What is machine learning?"

    chunks = retrieve_chunks(question)

    print("=" * 80)

    for i, chunk in enumerate(chunks, start=1):

        print(f"\nChunk {i}\n")

        print(chunk)

        print("-" * 80)


if __name__ == "__main__":
    main()