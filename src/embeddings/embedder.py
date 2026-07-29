import json
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer



def load_chunks(input_file: Path) -> list[str]:

    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = []

    for chunk in data:
        texts.append(chunk["text"])

    return texts


def save_embeddings(embeddings, output_file: Path) -> None:

    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_file, embeddings)
    ...


def main():

    input_file = Path(
        "data/processed/lecture01_chunks.json"
    )

    output_file = Path(
        "data/embeddings/lecture01_embeddings.npy"
    )

    texts = load_chunks(input_file)

    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5"
    )

    embeddings = model.encode(texts)
    

    print(type(embeddings))
    print(embeddings.shape)

    save_embeddings(
        embeddings,
        output_file
    )

    print(f"Saved embeddings to: {output_file}")


