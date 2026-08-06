from typing import Tuple

MAX_QUESTION_LENGTH = 500

BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "system prompt",
    "developer message",
    "jailbreak",
    "forget previous instructions",
]


def validate_question(question: str) -> Tuple[bool, str]:

    question = question.strip()

    if len(question) == 0:
        return False, "Please enter a question."

    if len(question) > MAX_QUESTION_LENGTH:
        return False, "Question is too long."

    lower = question.lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern in lower:
            return (
                False,
                "This request isn't supported."
            )

    return True, ""