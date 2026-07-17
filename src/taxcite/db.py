"""Database layer: pgvector schema, upsert, and cosine-similarity search."""
from __future__ import annotations

import os

import numpy as np
import psycopg2
import psycopg2.extensions
from pgvector.psycopg2 import register_vector

from taxcite.chunk import Chunk

_MIGRATION = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    pub_id      TEXT    NOT NULL,
    ordinal     INTEGER NOT NULL,
    first_page  INTEGER NOT NULL,
    last_page   INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    embedding   vector(1024),
    UNIQUE (pub_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


def get_connection() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    # register_vector needs the vector type to exist first, so a brand-new
    # database (no prior run_migration call) would otherwise fail to connect.
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    return conn


def run_migration(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(_MIGRATION)
    conn.commit()


def upsert_chunk(
    conn: psycopg2.extensions.connection,
    chunk: Chunk,
    embedding: list[float],
) -> None:
    vec = np.array(embedding, dtype=np.float32)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks (pub_id, ordinal, first_page, last_page, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (pub_id, ordinal) DO UPDATE SET
                first_page = EXCLUDED.first_page,
                last_page  = EXCLUDED.last_page,
                text       = EXCLUDED.text,
                embedding  = EXCLUDED.embedding
            """,
            (chunk.pub_id, chunk.ordinal, chunk.first_page, chunk.last_page, chunk.text, vec),
        )
    conn.commit()


def prune_chunks(conn: psycopg2.extensions.connection, pub_id: str, keep_count: int) -> int:
    """Drop orphan rows left when a re-ingest yields fewer chunks than before.

    Ordinals are positional (0..keep_count-1), so upsert overwrites the current
    range but never touches higher ordinals from a prior, longer run. Those
    stale rows keep their old embeddings and would pollute search. keep_count=0
    clears the publication entirely.
    """
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM chunks WHERE pub_id = %s AND ordinal >= %s",
            (pub_id, keep_count),
        )
        deleted = cur.rowcount
    conn.commit()
    return deleted


def search_chunks(
    conn: psycopg2.extensions.connection,
    embedding: list[float],
    top_k: int = 8,
    pub_ids: list[str] | None = None,
) -> list[Chunk]:
    vec = np.array(embedding, dtype=np.float32)
    with conn.cursor() as cur:
        if pub_ids:
            cur.execute(
                """
                SELECT pub_id, ordinal, first_page, last_page, text
                FROM chunks
                WHERE pub_id = ANY(%s)
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (pub_ids, vec, top_k),
            )
        else:
            cur.execute(
                """
                SELECT pub_id, ordinal, first_page, last_page, text
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (vec, top_k),
            )
        rows = cur.fetchall()
    return [
        Chunk(pub_id=r[0], ordinal=r[1], first_page=r[2], last_page=r[3], text=r[4])
        for r in rows
    ]


def count_chunks(conn: psycopg2.extensions.connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pub_id, COUNT(*) FROM chunks GROUP BY pub_id ORDER BY pub_id"
        )
        return {row[0]: int(row[1]) for row in cur.fetchall()}
