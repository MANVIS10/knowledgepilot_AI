import json
from pathlib import Path

from src.preprocessing.html_parser import extract_text_from_html


def split_into_words(text: str) -> list[str]:
    return text.split()


def create_chunks(
    words: list[str],
    chunk_size: int = 300,
    overlap: int = 50,
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
    output_file: Path,
):

    output_data = []

    for i, chunk in enumerate(chunks, start=1):
        output_data.append(
            {
                "chunk_id": i,
                "text": chunk
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            output_data,
            f,
            indent=4,
            ensure_ascii=False
        )


def process_lecture(
    html_file: Path,
    output_file: Path,
):

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

    print("=" * 60)
    print(f"Processed : {html_file.name}")
    print(f"Words      : {len(words)}")
    print(f"Chunks     : {len(chunks)}")
    print(f"Saved to   : {output_file}")


def main():

    transcript_folder = Path("data/raw/transcripts")

    output_folder = Path("data/processed")

    output_folder.mkdir(parents=True, exist_ok=True)

    html_files = sorted(transcript_folder.glob("*.html"))

    print(f"\nFound {len(html_files)} transcript files.\n")

    for html_file in html_files:

        lecture_name = html_file.stem.replace("_transcript", "")

        output_file = output_folder / f"{lecture_name}_chunks.json"

        process_lecture(
            html_file,
            output_file
        )

    print("\nAll lectures processed successfully!\n")


if __name__ == "__main__":
    main()