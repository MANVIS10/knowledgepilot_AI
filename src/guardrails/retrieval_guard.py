SIMILARITY_THRESHOLD = 0.60


def validate_retrieval(question, retrieved_chunks):

    if not retrieved_chunks:
        return (
            False,
            "I couldn't find relevant information."
        )

    best_distance = retrieved_chunks[0]["distance"]

    if best_distance > SIMILARITY_THRESHOLD:
        return (
            False,
            "I couldn't find relevant information in the lecture material."
        )

    return True, ""