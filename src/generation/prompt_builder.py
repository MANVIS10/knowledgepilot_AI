def build_prompt(
    question: str,
    chunks: list[str],
    history: list[dict]
) -> str:
    """
    Build a prompt for the LLM using retrieved chunks
    and conversation history.

    Args:
        question (str): User's question.
        chunks (list[str]): Retrieved chunks.
        history (list[dict]): Conversation history.

    Returns:
        str: Prompt for the LLM.
    """

    # -----------------------------
    # Build conversation history
    # -----------------------------
    history_text = ""

    for message in history:
        role = message["role"].capitalize()
        content = message["content"]
        history_text += f"{role}: {content}\n"

    # -----------------------------
    # Build context
    # -----------------------------
    context = "\n\n".join(chunks)

    # -----------------------------
    # Prompt
    # -----------------------------
    prompt = f"""
You are a Stanford CS229 Teaching Assistant.

Use the conversation history to understand follow-up questions.

Answer ONLY using the context provided below.

If the answer is not present in the context, reply exactly:

"I don't have enough information in the provided context."

Conversation History
--------------------------------------------------
{history_text}

Retrieved Context
--------------------------------------------------
{context}

Question
--------------------------------------------------
{question}

Answer:
"""

    return prompt


if __name__ == "__main__":

    history = [
        {
            "role": "user",
            "content": "What is machine learning?"
        },
        {
            "role": "assistant",
            "content": "Machine learning is the field that enables computers to learn from data."
        }
    ]

    chunks = [
        "Supervised learning learns from labeled data.",
        "Unsupervised learning discovers hidden patterns in unlabeled data."
    ]

    question = "What is supervised learning?"

    prompt = build_prompt(
        question,
        chunks,
        history
    )

    print(prompt)