from src.retrieval.search import retrieve_chunks
from src.retrieval.query_rewriter import rewrite_query
from src.generation.generator import Generator
from src.guardrails.input_guard import validate_question
from src.guardrails.retrieval_guard import validate_retrieval
from src.guardrails.output_guard import validate_output
from src.memory.conversation_memory import ConversationMemory
from src.config import USE_OUTPUT_GUARD, MAX_HISTORY
from src.logging.logger import logger

from collections import OrderedDict
from threading import Lock

# Upper bound on remembered sessions so memory use can't grow forever
MAX_SESSIONS = 1000


class KnowledgeBase:

    def __init__(self):

        self.generator = Generator()
        self._memories = OrderedDict()
        self._lock = Lock()

    ############################################################
    # PER-SESSION MEMORY
    ############################################################

    def get_memory(self, session_id: str | None) -> ConversationMemory:
        """
        Return the conversation memory that belongs to one session.
        A missing session_id gets a throwaway memory (stateless request).
        """

        if session_id is None:
            return ConversationMemory(MAX_HISTORY)

        with self._lock:
            memory = self._memories.get(session_id)

            if memory is None:
                memory = ConversationMemory(MAX_HISTORY)
                self._memories[session_id] = memory
                if len(self._memories) > MAX_SESSIONS:
                    self._memories.popitem(last=False)
            else:
                self._memories.move_to_end(session_id)

            return memory

    def clear_memory(self, session_id: str | None):

        if session_id is None:
            return

        with self._lock:
            self._memories.pop(session_id, None)

    ############################################################
    # NORMAL METHOD (FastAPI / Swagger)
    ############################################################

    def ask(
        self,
        question: str,
        top_k: int = 5,
        session_id: str | None = None
    ) -> dict:

        memory = self.get_memory(session_id)
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

        # Earlier turns only (the new question is saved after we answer)
        history = memory.get_messages()

        # Search with a standalone version of follow-up questions
        search_query = rewrite_query(question, history)

        # Retrieve chunks
        retrieved_chunks = retrieve_chunks(
            search_query,
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

        # Save the turn only once we have an answer, so blocked or
        # irrelevant questions never leave a dangling user message
        memory.add_user_message(question)
        memory.add_assistant_message(answer)

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
        top_k: int = 5,
        session_id: str | None = None
    ):

        memory = self.get_memory(session_id)

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

        history = memory.get_messages()
        search_query = rewrite_query(question, history)

        retrieved_chunks = retrieve_chunks(
            search_query,
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
                # The user has already read the streamed answer, so don't
                # make it vanish: keep it and add a visible warning below.
                # Memory keeps the original text, not the warning.
                yield {
                    "answer": f"{final_answer}\n\n---\n⚠️ {validated_answer}",
                    "sources": sources,
                    "retrieved_chunks": retrieved_chunks,
                    "passed_chunks": retrieved_chunks,
                    "success": False
                }

        memory.add_user_message(question)
        memory.add_assistant_message(final_answer)

