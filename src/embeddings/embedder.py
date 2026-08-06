import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


def load_chunks(input_file: Path):

    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = []

    for chunk in data:
        texts.append(chunk["text"])

    return texts


def save_embeddings(
    embeddings,
    output_file: Path,
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_file,
        embeddings
    )


def process_file(
    input_file: Path,
    output_file: Path,
):

    texts = load_chunks(input_file)

    embeddings = model.encode(texts)

    save_embeddings(
        embeddings,
        output_file
    )

    print(f"Processed {input_file.name}")
    print(embeddings.shape)
    print()


def main():

    processed_folder = Path("data/processed")

    embedding_folder = Path("data/embeddings")

    embedding_folder.mkdir(exist_ok=True)

    for chunk_file in sorted(processed_folder.glob("*_chunks.json")):

        output_file = embedding_folder / f"{chunk_file.stem.replace('_chunks','')}_embeddings.npy"

        process_file(
            chunk_file,
            output_file
        )


if __name__ == "__main__":
    main()