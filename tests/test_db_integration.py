"""Real-Postgres integration tests for the db layer.

Skipped when DATABASE_URL is not set (the CI default; no live Postgres there).
Run locally against a Postgres instance with pgvector enabled:

    DATABASE_URL=postgresql://user:pass@localhost/taxcite pytest tests/test_db_integration.py
"""
from __future__ import annotations

import os
import uuid

import pytest

from taxcite.chunk import Chunk

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set; requires a live Postgres instance with pgvector",
)

_DIM = 1024


@pytest.fixture
def live_conn():
    from taxcite.db import close_pool, get_connection, release_connection, run_migration
    conn = get_connection()
    run_migration(conn)
    yield conn
    release_connection(conn)
    # Tears the pool down (not just this connection) so later test modules in
    # the same pytest process don't inherit a live pool when they mock get_connection.
    close_pool()


def _pub() -> str:
    return f"inttest-{uuid.uuid4().hex[:8]}"


def test_upsert_and_search_round_trip(live_conn):
    from taxcite.db import search_chunks, upsert_chunk

    pub_id = _pub()
    chunk = Chunk(pub_id=pub_id, ordinal=0, first_page=1, last_page=1, text="Standard deduction amount.")
    embedding = [0.1] * _DIM

    upsert_chunk(live_conn, chunk, embedding)
    results = search_chunks(live_conn, embedding, top_k=5)

    assert any(r.pub_id == pub_id for r in results)


def test_upsert_is_idempotent(live_conn):
    from taxcite.db import count_chunks, upsert_chunk

    pub_id = _pub()
    chunk = Chunk(pub_id=pub_id, ordinal=0, first_page=1, last_page=1, text="original")
    embedding = [0.2] * _DIM

    upsert_chunk(live_conn, chunk, embedding)
    upsert_chunk(live_conn, chunk, embedding)

    counts = count_chunks(live_conn)
    assert counts.get(pub_id, 0) == 1


def test_prune_removes_stale_ordinals(live_conn):
    from taxcite.db import count_chunks, prune_chunks, upsert_chunk

    pub_id = _pub()
    embedding = [0.3] * _DIM

    for i in range(3):
        upsert_chunk(
            live_conn,
            Chunk(pub_id=pub_id, ordinal=i, first_page=i + 1, last_page=i + 1, text=f"chunk {i}"),
            embedding,
        )

    deleted = prune_chunks(live_conn, pub_id, keep_count=1)

    assert deleted == 2
    assert count_chunks(live_conn).get(pub_id, 0) == 1
