from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

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
