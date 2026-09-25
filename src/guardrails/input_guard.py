import re
import unicodedata
from typing import Tuple

MAX_QUESTION_LENGTH = 500

# Digits/symbols attackers swap in for letters ("ign0re" -> "ignore")
LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s",
})

# Matched against the NORMALIZED text (see normalize below)
BLOCKED_PATTERNS = [
    # "ignore / disregard / forget ... previous / all / prior ... instructions"
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,30}\b(previous|prior|above|earlier|all|your)\b.{0,20}\b(instructions?|rules?|prompts?|guidelines?)\b",
    # asking for the hidden prompt
    r"\b(system|developer|hidden|initial)\s+(prompt|message|instructions?)\b",
    r"\b(reveal|show|print|display|repeat)\b.{0,20}\b(your|the)\b.{0,15}\b(prompt|rules|instructions)\b",
    r"\brepeat\b.{0,30}\b(text|words|everything|instructions?)\b.{0,20}\b(above|before)\b",
    r"\bwhat (were|are) you (told|instructed)\b",
    # jailbreak phrasing
    r"\bjailbreak\b",
    r"\b(act as dan|do anything now)\b",
    r"\bpretend\b.{0,30}\b(no|without)\b.{0,15}\b(rules?|restrictions?|limits?|guidelines?)\b",
]

_COMPILED = [re.compile(p) for p in BLOCKED_PATTERNS]


def normalize(text: str) -> str:
    """Lower-case, undo look-alike characters and leetspeak, drop punctuation,
    and collapse whitespace so simple obfuscation can't dodge the patterns."""

    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(LEET_MAP)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def validate_question(question: str) -> Tuple[bool, str]:

    question = question.strip()

    if len(question) == 0:
        return False, "Please enter a question."

    if len(question) > MAX_QUESTION_LENGTH:
        return False, "Question is too long."

    normalized = normalize(question)

    for pattern in _COMPILED:
        if pattern.search(normalized):
            return (
                False,
                "This request isn't supported."
            )

    return True, ""
