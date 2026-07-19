from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from taxcite import db as db_module
from taxcite.chunk import Chunk
from taxcite.db import count_chunks, prune_chunks, run_migration, search_chunks, upsert_chunk

CHUNK_A = Chunk(pub_id="p501", ordinal=0, first_page=1, last_page=2, text="standard deduction text")
EMBEDDING_A = [0.1] * 1024


def _make_conn(rows=None):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = lambda s: cursor
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.fetchall.return_value = rows or []
    conn.cursor.return_value = cursor
    return conn, cursor


@pytest.fixture(autouse=True)
def _reset_pool_singleton():
    """The connection pool is a module-level singleton; isolate tests from it."""
    db_module._pool = None
    yield
    db_module._pool = None


def test_run_migration_executes_sql():
    conn, cursor = _make_conn()
    run_migration(conn)
    sql = cursor.execute.call_args[0][0]
    assert "CREATE TABLE IF NOT EXISTS chunks" in sql
    assert "vector(1024)" in sql
    conn.commit.assert_called_once()


def test_upsert_chunk_passes_numpy_array():
    conn, cursor = _make_conn()
    upsert_chunk(conn, CHUNK_A, EMBEDDING_A)
    args = cursor.execute.call_args[0][1]
    assert args[0] == "p501"
    assert args[1] == 0
    assert isinstance(args[5], np.ndarray)
    assert args[5].shape == (1024,)
    conn.commit.assert_called_once()


def test_upsert_chunk_skips_commit_when_batching():
    conn, cursor = _make_conn()
    upsert_chunk(conn, CHUNK_A, EMBEDDING_A, commit=False)
    conn.commit.assert_not_called()


def test_search_chunks_returns_chunk_list():
    rows = [("p501", 0, 1, 2, "standard deduction text")]
    conn, cursor = _make_conn(rows=rows)
    result = search_chunks(conn, EMBEDDING_A, top_k=3)
    assert len(result) == 1
    assert result[0] == CHUNK_A


def test_search_chunks_with_pub_filter():
    conn, cursor = _make_conn(rows=[])
    search_chunks(conn, EMBEDDING_A, top_k=3, pub_ids=["p501"])
    sql = cursor.execute.call_args[0][0]
    assert "pub_id = ANY" in sql


def test_count_chunks_returns_dict():
    rows = [("p501", 42), ("p590a", 18)]
    conn, cursor = _make_conn(rows=rows)
    result = count_chunks(conn)
    assert result == {"p501": 42, "p590a": 18}


def test_prune_chunks_deletes_orphan_ordinals():
    conn, cursor = _make_conn()
    cursor.rowcount = 3
    deleted = prune_chunks(conn, "p501", keep_count=10)
    sql, params = cursor.execute.call_args[0]
    assert "DELETE FROM chunks" in sql
    assert "ordinal >= %s" in sql
    assert params == ("p501", 10)
    assert deleted == 3
    conn.commit.assert_called_once()


def test_prune_chunks_skips_commit_when_batching():
    conn, cursor = _make_conn()
    cursor.rowcount = 3
    prune_chunks(conn, "p501", keep_count=10, commit=False)
    conn.commit.assert_not_called()


# connection pool


def test_get_connection_creates_pool_once_and_acquires_from_it(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/taxcite")
    conn, _ = _make_conn()
    pool_instance = MagicMock()
    pool_instance.getconn.return_value = conn
    with (
        patch("taxcite.db.psycopg2.pool.ThreadedConnectionPool", return_value=pool_instance) as MockPool,
        patch("taxcite.db.register_vector") as mock_register,
    ):
        result = db_module.get_connection()
        db_module.get_connection()

    MockPool.assert_called_once_with(1, 10, "postgresql://user:pass@localhost/taxcite")
    assert pool_instance.getconn.call_count == 2
    assert result is conn
    mock_register.assert_called_with(conn)


def test_release_connection_returns_conn_to_pool_instead_of_closing(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/taxcite")
    conn, _ = _make_conn()
    pool_instance = MagicMock()
    pool_instance.getconn.return_value = conn
    with (
        patch("taxcite.db.psycopg2.pool.ThreadedConnectionPool", return_value=pool_instance),
        patch("taxcite.db.register_vector"),
    ):
        acquired = db_module.get_connection()
        db_module.release_connection(acquired)

    pool_instance.putconn.assert_called_once_with(conn)
    conn.close.assert_not_called()


def test_release_connection_without_a_pool_closes_directly():
    conn = MagicMock()
    db_module.release_connection(conn)
    conn.close.assert_called_once()


def test_close_pool_closes_all_connections_and_clears_singleton(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/taxcite")
    conn, _ = _make_conn()
    pool_instance = MagicMock()
    pool_instance.getconn.return_value = conn
    with (
        patch("taxcite.db.psycopg2.pool.ThreadedConnectionPool", return_value=pool_instance),
        patch("taxcite.db.register_vector"),
    ):
        db_module.get_connection()
        db_module.close_pool()

    pool_instance.closeall.assert_called_once()
    assert db_module._pool is None
