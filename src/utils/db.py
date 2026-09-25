import os
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

# Connection details must come from the environment (.env locally, the
# host's settings in production). Never hard-code them here.
REQUIRED_VARS = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]


def get_connection():
    missing = [name for name in REQUIRED_VARS if not os.getenv(name)]

    if missing:
        raise RuntimeError(
            "Missing database settings: "
            + ", ".join(missing)
            + ". Set them in your .env file (see .env.example)."
        )

    conn = psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        database=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        # "prefer" works for a local Docker database (no SSL) and still uses
        # SSL when the server offers it. Set DB_SSLMODE=require for hosted DBs.
        sslmode=os.getenv("DB_SSLMODE", "prefer"),
    )

    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    except Exception:
        conn.rollback()

    register_vector(conn)
    return conn
