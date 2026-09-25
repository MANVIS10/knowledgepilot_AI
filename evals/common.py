import json
import re
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
REPORT_DIR = EVALS_DIR / "reports"


def load_cases() -> dict:
    with open(EVALS_DIR / "datasets" / "component_cases.json", encoding="utf-8") as f:
        return json.load(f)


def is_relevant(chunk_text: str, relevant_regex: str) -> bool:
    """A chunk counts as relevant if it contains the topic's keywords."""
    return re.search(relevant_regex, chunk_text, re.I) is not None


def save_report(name: str, data: dict) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    with open(REPORT_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
