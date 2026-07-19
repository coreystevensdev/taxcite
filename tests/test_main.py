"""Tests for the CLI ingest command's transaction batching."""
from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import psycopg2
import pytest

from taxcite.__main__ import cmd_ingest
from taxcite.chunk import Chunk
from taxcite.manifest import Publication
from taxcite.parse import Page

_PUB = Publication("p501", "Dependents, Standard Deduction, and Filing Information")
_CHUNKS = [
    Chunk(pub_id="p501", ordinal=0, first_page=1, last_page=1, text="a"),
    Chunk(pub_id="p501", ordinal=1, first_page=2, last_page=2, text="b"),
    Chunk(pub_id="p501", ordinal=2, first_page=3, last_page=3, text="c"),
]


def _patch_ingest_pipeline(stack: ExitStack, conn: MagicMock) -> None:
    """Stub every step of cmd_ingest except the db calls under test."""
    stack.enter_context(patch("taxcite.__main__.get_publication", return_value=_PUB))
    stack.enter_context(patch("taxcite.__main__.fetch_publication", return_value="ignored-path"))
    stack.enter_context(patch("taxcite.__main__.parse_pdf", return_value=[Page(number=1, text="x")]))
    stack.enter_context(patch("taxcite.__main__.chunk_pages", return_value=_CHUNKS))
    stack.enter_context(patch("taxcite.embed.embed_texts", return_value=[[0.0] * 1024] * len(_CHUNKS)))
    stack.enter_context(patch("taxcite.db.get_connection", return_value=conn))
    stack.enter_context(patch("taxcite.db.release_connection"))
    stack.enter_context(patch("taxcite.db.run_migration"))


def test_ingest_commits_once_per_publication_not_per_chunk():
    conn = MagicMock()
    with ExitStack() as stack:
        _patch_ingest_pipeline(stack, conn)
        mock_upsert = stack.enter_context(patch("taxcite.db.upsert_chunk"))
        mock_prune = stack.enter_context(patch("taxcite.db.prune_chunks"))

        status = cmd_ingest(["p501"])

    assert status == 0
    assert mock_upsert.call_count == len(_CHUNKS)
    for call in mock_upsert.call_args_list:
        assert call.kwargs["commit"] is False
    mock_prune.assert_called_once()
    assert mock_prune.call_args.kwargs["commit"] is False
    conn.commit.assert_called_once()
    conn.rollback.assert_not_called()


def test_ingest_rolls_back_the_whole_publication_on_db_error():
    conn = MagicMock()
    with ExitStack() as stack:
        _patch_ingest_pipeline(stack, conn)
        stack.enter_context(patch("taxcite.db.upsert_chunk"))
        stack.enter_context(
            patch("taxcite.db.prune_chunks", side_effect=psycopg2.OperationalError("connection reset"))
        )

        with pytest.raises(psycopg2.OperationalError):
            cmd_ingest(["p501"])

    conn.rollback.assert_called_once()
    conn.commit.assert_not_called()
