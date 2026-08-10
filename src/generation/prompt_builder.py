def build_prompt(
    question: str,
    chunks: list,
    history: list[dict] = None
) -> str:
    """
    Build a prompt for the LLM using retrieved chunks
    and conversation history.

    Args:
        question (str): User's question.
        chunks (list): Retrieved chunks (can be list of dicts or list of strings).
        history (list[dict]): Conversation history.

    Returns:
        str: Prompt for the LLM.
    """
    if history is None:
        history = []

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
    formatted_chunks = []
    for i, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            lecture_name = chunk.get("lecture", "Unknown")
            import re
            match = re.search(r'\d+', lecture_name)
            if match:
                lecture_num = int(match.group())
                formatted_source = f"Lecture {lecture_num}, Section {chunk.get('chunk_id', i)}"
            else:
                formatted_source = f"{lecture_name.capitalize()}, Section {chunk.get('chunk_id', i)}"
            
            chunk_text = chunk.get("text", "")
            formatted_chunks.append(f"[{i}] Source: {formatted_source}\nContent: {chunk_text}")
        else:
            formatted_chunks.append(f"[{i}] Content: {chunk}")

    context = "\n\n".join(formatted_chunks)

    # -----------------------------
    # Prompt
    # -----------------------------
    prompt = f"""
You are a Stanford CS229 Teaching Assistant.

Use the conversation history to understand follow-up questions.

Answer the question using ONLY the retrieved context provided below. For every fact, claim, or statement you make in your answer, you MUST cite the source by appending its corresponding chunk number in square brackets, e.g. [1] or [2]. Place these citations at the end of the sentence or clause containing the information. If multiple sources support a statement, include multiple citations like [1][2]. Keep the answer clean and professional.

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
        {"lecture": "lecture01", "chunk_id": 5, "text": "Supervised learning learns from labeled data."},
        {"lecture": "lecture02", "chunk_id": 12, "text": "Unsupervised learning discovers hidden patterns in unlabeled data."}
    ]

    question = "What is supervised learning?"

    prompt = build_prompt(
        question,
        chunks,
        history
    )

    print(prompt)