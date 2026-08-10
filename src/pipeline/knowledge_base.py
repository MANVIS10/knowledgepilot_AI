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
                "sources": [],
                "retrieved_chunks": [],
                "passed_chunks": [],
                "success": False
            }

        # Retrieve chunks
        retrieved_chunks = retrieve_chunks(
            question,
            top_k
        )
        logger.info(
            f"Retrieved {len(retrieved_chunks)} chunks"
        )

        # Compute similarity scores
        for chunk in retrieved_chunks:
            chunk["similarity_score"] = max(0.0, min(1.0, 1.0 - chunk["distance"]))

        allowed, message = validate_retrieval(
            question,
            retrieved_chunks
        )

        if not allowed:
            return {
                "answer": message,
                "sources": [],
                "retrieved_chunks": retrieved_chunks,
                "passed_chunks": [],
                "success": False
            }

        # Extract chunk texts for guardrails
        chunk_texts = [
            chunk["text"]
            for chunk in retrieved_chunks
        ]

        history = self.memory.get_messages()

        # Generate answer using structured retrieved_chunks to build citations
        answer = self.generator.generate(
            question,
            retrieved_chunks,
            history
        )
        logger.info(
            f"Generated answer ({len(answer)} characters)"
        )

        # Output guard
        if USE_OUTPUT_GUARD:
            allowed_out, validated_answer = validate_output(
                question,
                answer,
                chunk_texts
            )
            if not allowed_out:
                answer = validated_answer
                allowed = False

        self.memory.add_assistant_message(answer)

        sources = [
            {
                "lecture": chunk["lecture"].replace("lecture", "Lecture "),
                "chunk_id": chunk["chunk_id"],
                "similarity_score": chunk["similarity_score"],
                "text": chunk["text"]
            }
            for chunk in retrieved_chunks
        ]
        logger.info(
            f"Sources: {[s['lecture'] for s in sources]}"
        )

        if not allowed:
            logger.warning(message)
            return {
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": retrieved_chunks,
                "passed_chunks": retrieved_chunks,
                "success": False
            }

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "passed_chunks": retrieved_chunks,
            "success": True
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
            yield {
                "answer": message,
                "sources": [],
                "retrieved_chunks": [],
                "passed_chunks": [],
                "success": False
            }
            return

        retrieved_chunks = retrieve_chunks(
            question,
            top_k
        )

        # Compute similarity scores
        for chunk in retrieved_chunks:
            chunk["similarity_score"] = max(0.0, min(1.0, 1.0 - chunk["distance"]))

        allowed, message = validate_retrieval(
            question,
            retrieved_chunks
        )

        if not allowed:
            yield {
                "answer": message,
                "sources": [],
                "retrieved_chunks": retrieved_chunks,
                "passed_chunks": [],
                "success": False
            }
            return

        chunk_texts = [
            chunk["text"]
            for chunk in retrieved_chunks
        ]

        history = self.memory.get_messages()

        sources = [
            {
                "lecture": chunk["lecture"].replace("lecture", "Lecture "),
                "chunk_id": chunk["chunk_id"],
                "similarity_score": chunk["similarity_score"],
                "text": chunk["text"]
            }
            for chunk in retrieved_chunks
        ]

        # Yield initially with empty answer but complete retrieval metadata
        yield {
            "answer": "",
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "passed_chunks": retrieved_chunks,
            "success": True
        }

        final_answer = ""

        for partial in self.generator.stream_generate(
            question,
            retrieved_chunks,
            history
        ):
            final_answer = partial
            yield {
                "answer": partial,
                "sources": sources,
                "retrieved_chunks": retrieved_chunks,
                "passed_chunks": retrieved_chunks,
                "success": True
            }

        # Apply output guard at the end of streaming
        if USE_OUTPUT_GUARD:
            allowed_out, validated_answer = validate_output(
                question,
                final_answer,
                chunk_texts
            )
            if not allowed_out:
                final_answer = validated_answer
                yield {
                    "answer": final_answer,
                    "sources": sources,
                    "retrieved_chunks": retrieved_chunks,
                    "passed_chunks": retrieved_chunks,
                    "success": False
                }

        self.memory.add_assistant_message(final_answer)
