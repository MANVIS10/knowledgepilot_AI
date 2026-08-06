from pathlib import Path

from bs4 import BeautifulSoup


def extract_text_from_html(file_path: Path) -> str:

    with open(file_path, "r", encoding="windows-1252") as file:
        html = file.read()

    soup = BeautifulSoup(html, "html.parser")

    text = soup.get_text()

    text = " ".join(text.split())

    return text