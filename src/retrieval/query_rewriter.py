from dotenv import load_dotenv
from openai import OpenAI

from src.config import MODEL_NAME, REWRITE_FOLLOWUPS
from src.logging.logger import logger

load_dotenv()

client = OpenAI()

MAX_REWRITE_CHARS = 300
MAX_HISTORY_CHARS = 300  # per message, keeps the extra call small and cheap


def rewrite_query(question: str, history: list[dict]) -> str:
    """
    Turn a follow-up such as "Why does it need a learning rate?" into a
    standalone search query ("Why does gradient descent need a learning rate?").

    Only the SEARCH uses the rewritten text; the user's own words still go into
    the prompt. Any problem falls back to the original question, so this can
    never make an answer fail.
    """

    if not REWRITE_FOLLOWUPS or not history:
        return question

    conversation = "\n".join(
        f"{message['role'].capitalize()}: {message['content'][:MAX_HISTORY_CHARS]}"
        for message in history
    )

    prompt = f"""Rewrite the user's latest question as a standalone search query, using the conversation only to resolve words like "it", "that" or "this".

If the question is already standalone, return it unchanged.
Return ONLY the rewritten question, with no explanation.

Conversation:
{conversation}

Latest question: {question}

Standalone question:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80,
        )
        rewritten = response.choices[0].message.content.strip().strip('"\'')
    except Exception as error:
        logger.warning(f"Query rewrite failed, using original question: {error}")
        return question

    if not rewritten or len(rewritten) > MAX_REWRITE_CHARS:
        return question

    logger.info(f"Rewrote query: {question!r} -> {rewritten!r}")
    return rewritten
