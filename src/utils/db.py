import os
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "altaria.proxy.rlwy.net"),
        port=int(os.getenv("DB_PORT", "37785")),
        database=os.getenv("DB_NAME", "railway"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "YViFfTrfMoTWnxgLduCBcHbwcJYRknDM"),
        sslmode=os.getenv("DB_SSLMODE", "require"),
    )

    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    except Exception:
        conn.rollback()

    register_vector(conn)
    return conn
