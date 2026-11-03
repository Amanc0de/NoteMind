import os
import psycopg


def get_connection():
    database_url = os.getenv("DATABASE_URL")

    return psycopg.connect(database_url)


def create_tables():
    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id),
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

        conn.commit()
def save_document(filename):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename)
                VALUES (%s)
                RETURNING id;
                """,
                (filename,)
            )

            document_id = cur.fetchone()[0]

        conn.commit()

    return document_id


def save_chat(document_id, question, answer):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_history
                (document_id, question, answer)
                VALUES (%s, %s, %s);
                """,
                (document_id, question, answer)
            )

        conn.commit()


def get_chat_history(document_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT question, answer, created_at
                FROM chat_history
                WHERE document_id = %s
                ORDER BY created_at ASC;
                """,
                (document_id,)
            )

            history = cur.fetchall()

    return history