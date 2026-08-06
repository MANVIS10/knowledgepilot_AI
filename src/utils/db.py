import psycopg2
from pgvector.psycopg2 import register_vector

def get_connection():
    conn = psycopg2.connect(
        host="altaria.proxy.rlwy.net",
        port=37785,
        database="railway",
        user="postgres",
        password="YViFfTrfMoTWnxgLduCBcHbwcJYRknDM",
        sslmode="require",
    )

    register_vector(conn)
    return conn