from openai import OpenAI
from src.config import MODEL_NAME

client = OpenAI()

# The exact refusal the prompt tells the model to give when the context has no answer
NO_ANSWER_TEXT = "I don't have enough information in the provided context."


def validate_output(
    question: str,
    answer: str,
    chunks: list[str]
) -> tuple[bool, str]:
    """
    Validate that the generated answer is grounded
    in the retrieved context.
    """

    if not answer or len(answer.strip()) == 0:
        return False, "The model returned an empty response."

    # An honest "I don't know" makes no claims, so there is nothing to verify
    if answer.strip().strip("\"'") == NO_ANSWER_TEXT:
        return True, answer

    context = "\n\n".join(chunks)

    prompt = f"""
You are evaluating a Retrieval-Augmented Generation system.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Determine whether the answer is fully supported by the retrieved context.

Respond with ONLY one word:

SUPPORTED

or

UNSUPPORTED
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # Tolerate case and punctuation ("Supported.", "**SUPPORTED**").
    # UNSUPPORTED contains SUPPORTED, so it must be checked by prefix, not substring.
    decision = response.choices[0].message.content.strip().strip(" *\"'`.").upper()

    if decision.startswith("SUPPORTED"):
        return True, answer

    return (
        False,
        "I couldn't verify this answer using the retrieved lecture material."
    )