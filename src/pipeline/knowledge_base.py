from src.retrieval.search import retrieve_chunks
from src.generation.generator import Generator
from src.guardrails.input_guard import validate_question
from src.guardrails.retrieval_guard import validate_retrieval
from src.guardrails.output_guard import validate_output
from src.memory.conversation_memory import ConversationMemory
from src.config import USE_OUTPUT_GUARD
from src.logging.logger import logger


class KnowledgeBase:

    def __init__(self):

        self.generator = Generator()
        self.memory = ConversationMemory()
    ############################################################
    # NORMAL METHOD (FastAPI / Swagger)
    ############################################################

    def ask(
        self,
        question: str,
        top_k: int = 5
    ) -> dict:

        self.memory.add_user_message(question)
        logger.info(f"Question: {question}")

        allowed, message = validate_question(question)

        if not allowed:
            return {
                "answer": message,
                "sources": []
            }

        # Retrieve chunks
        retrieved_chunks = retrieve_chunks(
            question,
            top_k
        )
        logger.info(
        f"Retrieved {len(retrieved_chunks)} chunks"
)

        allowed, message = validate_retrieval(
            question,
            retrieved_chunks
        )

        if not allowed:
            return {
                "answer": message,
                "sources": []
            }

        # Extract chunk texts
        chunk_texts = [
            chunk["text"]
            for chunk in retrieved_chunks
        ]

        history = self.memory.get_messages()

        # Generate answer
        answer = self.generator.generate(
            question,
            chunk_texts,
            history
        )
        logger.info(
        f"Generated answer ({len(answer)} characters)"
          )

        # Output guard
        if USE_OUTPUT_GUARD:

         allowed, validated_answer = validate_output(
        question,
        answer,
        chunk_texts
    )
        if not allowed :
         answer = validated_answer

        if not allowed:
            answer = validated_answer

        self.memory.add_assistant_message(answer)

        sources = [
            {
                "lecture": chunk["lecture"].replace("lecture", "Lecture "),
            }
            for chunk in retrieved_chunks
        ]
        logger.info(
    f"Sources: {[s['lecture'] for s in sources]}"
)
        if not allowed:
         logger.warning(message)

         return {
        "answer": message,
        "sources": []
    }
        return {
            "answer": answer,
            "sources": sources
        }

    ############################################################
    # STREAMING METHOD (Gradio Only)
    ############################################################

    def stream_answer(
        self,
        question: str,
        top_k: int = 5
    ):

        self.memory.add_user_message(question)

        allowed, message = validate_question(question)

        if not allowed:
            yield message
            return

        retrieved_chunks = retrieve_chunks(
            question,
            top_k
        )

        allowed, message = validate_retrieval(
            question,
            retrieved_chunks
        )

        if not allowed:
            yield message
            return

        chunk_texts = [
            chunk["text"]
            for chunk in retrieved_chunks
        ]

        history = self.memory.get_messages()

        final_answer = ""

        for partial in self.generator.stream_generate(
            question,
            chunk_texts,
            history
        ):
            final_answer = partial
            yield partial

        self.memory.add_assistant_message(final_answer)