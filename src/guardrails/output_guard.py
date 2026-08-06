from openai import OpenAI

client = OpenAI()


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

    response = client.responses.create(
        model="gpt-5-nano",
        input=prompt
    )

    decision = response.output_text.strip()

    if decision == "SUPPORTED":
        return True, answer

    return (
        False,
        "I couldn't verify this answer using the retrieved lecture material."
    )