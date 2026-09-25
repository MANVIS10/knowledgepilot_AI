from src.config import SIMILARITY_THRESHOLD


def validate_retrieval(question, retrieved_chunks):

    if not retrieved_chunks:
        return (
            False,
            "I couldn't find relevant information."
        )

    best_distance = retrieved_chunks[0]["distance"]
    best_similarity = 1 - best_distance

    if best_similarity < SIMILARITY_THRESHOLD:
        return (
            False,
            "I couldn't find relevant information in the lecture material."
        )

    return True, ""