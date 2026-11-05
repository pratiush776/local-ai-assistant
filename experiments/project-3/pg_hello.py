import psycopg

DSN = "postgresql://postgres:rag@localhost:5432/postgres"

with psycopg.connect(DSN) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    version = conn.execute("SELECT version()").fetchone()
    version = version[0] if version else []
    has_vector = conn.execute(
        "SELECT extname FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    print("connected to:", version[:30])
    print("pgvector enabled:", has_vector)