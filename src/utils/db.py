import psycopg2
from pgvector.psycopg2 import register_vector

def get_connection():
    conn = psycopg2.connect(
        host="knowledgepilot-db",
        port=5432,
        database="knowledgepilot",
        user="postgres",
        password="postgres123"
    )

    register_vector(conn)
    return conn