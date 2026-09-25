from src.config import SIMILARITY_THRESHOLD


def validate_retrieval(question, retrieved_chunks):

    if not retrieved_chunks:
        return (
            False,
            "I couldn't find relevant information."
        )

    # Hybrid results are ordered by fused rank, not distance, so take the closest
    best_distance = min(chunk["distance"] for chunk in retrieved_chunks)
    best_similarity = 1 - best_distance

    if best_similarity < SIMILARITY_THRESHOLD:
        return (
            False,
            "I couldn't find relevant information in the lecture material."
        )

    return True, ""