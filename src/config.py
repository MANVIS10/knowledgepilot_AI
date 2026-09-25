MODEL_NAME = "gpt-4o-mini"

TOP_K = 5

MAX_HISTORY = 6

SIMILARITY_THRESHOLD = 0.60

USE_OUTPUT_GUARD = True

MAX_QUESTION_LENGTH = 500

# Rewrite follow-up questions ("why does it...") into standalone search queries
REWRITE_FOLLOWUPS = True

# Hybrid retrieval: vector search + keyword (full-text) search, merged by rank.
# OFF by default: measured worse than vector-only on follow-ups (see docs/plans).
HYBRID_SEARCH = False
HYBRID_CANDIDATES = 20   # how many results each search contributes before merging
RRF_K = 60               # Reciprocal Rank Fusion constant (60 is the usual default)
KEYWORD_MAX_DOC_FREQ = 0.08  # ignore query words found in more than 8% of chunks (too common to help)
KEYWORD_MAX_TERMS = 4        # keep at most this many of the rarest query words
