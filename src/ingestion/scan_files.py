from pathlib import Path
from bs4 import BeautifulSoup

transcript_folder = Path("data/raw/transcripts")
def extract_text_from_html(file: Path) -> str:
    html = file.read_text(encoding="windows-1252")
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    clean_text = " ".join(text.split())
    return clean_text

for file in transcript_folder.iterdir():

    if file.suffix == ".html":

        clean_text = extract_text_from_html(file)

        print(clean_text)

        