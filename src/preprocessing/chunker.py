import json
from pathlib import Path

from html_parser import extract_text_from_html


def split_into_words(text: str) -> list[str]:
    return text.split()


def create_chunks(
    words: list[str],
    chunk_size: int,
    overlap: int
) -> list[str]:

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]

        if chunk:
            chunks.append(" ".join(chunk))

    return chunks


def save_chunks(
    chunks: list[str],
    output_file: Path
) -> None:

    chunk_data = []

    for i, chunk in enumerate(chunks, start=1):
        chunk_data.append(
            {
                "chunk_id": i,
                "text": chunk
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            chunk_data,
            file,
            indent=4,
            ensure_ascii=False
        )


def main():

    html_file = Path(
        "data/raw/transcripts/lecture01_transcript.html"
    )

    output_file = Path(
        "data/processed/lecture01_chunks.json"
    )

    text = extract_text_from_html(html_file)

    words = split_into_words(text)

    chunks = create_chunks(
        words,
        chunk_size=300,
        overlap=50
    )

    save_chunks(
        chunks,
        output_file
    )

    print(f"Total words : {len(words)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Saved chunks to: {output_file}")


if __name__ == "__main__":
    main()