"""Tests for the FastAPI server endpoints."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import psycopg2
import pytest
from fastapi.testclient import TestClient

from taxcite.server import app

_BASE_STATE = {
    "question": "q",
    "chunks": [],
    "answer": "",
    "citations": [],
}

_VALID_THREAD_ID = "123e4567-e89b-12d3-a456-426614174000"


def _mock_graph(
    answer: str = "",
    citations: list | None = None,
    interrupted: bool = False,
    chunks: list | None = None,
) -> MagicMock:
    """Build a mock _graph that mimics LangGraph's invoke/get_state API."""
    mg = MagicMock()
    mg.invoke.return_value = None  # return value unused; server reads via get_state

    snapshot = MagicMock()
    if interrupted:
        snapshot.next = ("human_review",)
        snapshot.values = {**_BASE_STATE, "chunks": chunks or []}
    else:
        snapshot.next = ()
        snapshot.values = {**_BASE_STATE, "answer": answer, "citations": citations or []}

    mg.get_state.return_value = snapshot
    return mg


class TestHealth:
    def test_returns_200(self):
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestAsk:
    def test_returns_complete_answer_and_citations(self):
        citations = [{"pub_id": "p17", "first_page": 3, "last_page": 4}]
        mock = _mock_graph("Yes, it is deductible.", citations)
        with patch("taxcite.server._graph", mock):
            client = TestClient(app)
            resp = client.post("/ask", json={"question": "Is mortgage interest deductible?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "complete"
        assert body["answer"] == "Yes, it is deductible."
        assert body["citations"] == citations

    def test_returns_awaiting_review_when_graph_interrupted(self):
        from taxcite.chunk import Chunk

        fake_chunk = Chunk(pub_id="p936", ordinal=0, first_page=1, last_page=2, text="Mortgage interest rules.")
        mock = _mock_graph(interrupted=True, chunks=[fake_chunk])
        with patch("taxcite.server._graph", mock):
            client = TestClient(app)
            resp = client.post("/ask", json={"question": "Can I deduct interest?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "awaiting_review"
        assert "thread_id" in body
        assert len(body["chunks_preview"]) == 1
        assert body["chunks_preview"][0]["pub_id"] == "p936"

    def test_empty_question_returns_422(self):
        client = TestClient(app)
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_missing_question_returns_422(self):
        client = TestClient(app)
        resp = client.post("/ask", json={})
        assert resp.status_code == 422

    def test_agent_exception_returns_500(self):
        mock = MagicMock()
        mock.invoke.side_effect = RuntimeError("db unreachable")
        with patch("taxcite.server._graph", mock):
            client = TestClient(app)
            resp = client.post("/ask", json={"question": "test question"})

        assert resp.status_code == 500

    def test_cost_budget_exceeded_returns_503(self):
        from taxcite.cost import CostBudgetExceeded

        mock = MagicMock()
        mock.invoke.side_effect = CostBudgetExceeded("monthly-budget tripped")
        with patch("taxcite.server._graph", mock):
            client = TestClient(app)
            resp = client.post("/ask", json={"question": "test question"})

        assert resp.status_code == 503

    def test_database_unavailable_returns_503(self):
        import psycopg2

        mock = MagicMock()
        mock.invoke.side_effect = psycopg2.OperationalError("could not connect to server")
        with patch("taxcite.server._graph", mock):
            client = TestClient(app)
            resp = client.post("/ask", json={"question": "test question"})

        assert resp.status_code == 503
        assert "Database unavailable" in resp.json()["detail"]

    def test_db_query_error_returns_500(self):
        import psycopg2

        mock = MagicMock()
        mock.invoke.side_effect = psycopg2.ProgrammingError("relation does not exist")
        with patch("taxcite.server._graph", mock):
            client = TestClient(app)
            resp = client.post("/ask", json={"question": "test question"})

        assert resp.status_code == 500


class TestIngestThread:
    def test_records_failing_stage_and_cause(self):
        from taxcite import server

        with patch.dict(server._ingest, {}, clear=True), \
                patch("taxcite.db.get_connection") as mock_conn, \
                patch("taxcite.fetch.fetch_publication", side_effect=RuntimeError("HTTP 404 from IRS")):
            mock_conn.return_value = MagicMock()
            server._run_ingest_thread()

            assert server._ingest["state"] == "error"
            assert server._ingest["stage"]  # first pub in CORPUS
            assert "RuntimeError" in server._ingest["error"]
            assert "HTTP 404" in server._ingest["error"]

    def test_status_surfaces_error(self):
        from taxcite import server

        with patch.dict(server._ingest, {"state": "error", "stage": "p17", "error": "RuntimeError: boom"}, clear=True), \
                patch("taxcite.server.os.getenv", return_value=None):
            client = TestClient(app)
            resp = client.get("/status")

            assert resp.status_code == 200
            body = resp.json()
            assert body["ingest"] == "error"
            assert body["stage"] == "p17"
            # The raw exception cause stays in the log, not the public response.
            assert body["error"] == "ingest failed; see server logs"
            assert "RuntimeError" not in body["error"]

    def test_commits_once_per_publication_not_per_chunk(self):
        from taxcite import server
        from taxcite.chunk import Chunk
        from taxcite.manifest import Publication

        pub = Publication("p501", "Test Pub")
        chunks = [
            Chunk(pub_id="p501", ordinal=i, first_page=i + 1, last_page=i + 1, text=f"c{i}")
            for i in range(3)
        ]
        conn = MagicMock()

        with (
            patch.dict(server._ingest, {}, clear=True),
            patch("taxcite.manifest.CORPUS", (pub,)),
            patch("taxcite.fetch.fetch_publication", return_value="ignored-path"),
            patch("taxcite.parse.parse_pdf", return_value=[]),
            patch("taxcite.chunk.chunk_pages", return_value=chunks),
            patch("taxcite.embed.embed_texts", return_value=[[0.0] * 1024] * len(chunks)),
            patch("taxcite.db.get_connection", return_value=conn),
            patch("taxcite.db.release_connection"),
            patch("taxcite.db.upsert_chunk") as mock_upsert,
            patch("taxcite.db.prune_chunks") as mock_prune,
        ):
            server._run_ingest_thread()

            assert server._ingest["state"] == "ready"
            assert mock_upsert.call_count == len(chunks)
            for call in mock_upsert.call_args_list:
                assert call.kwargs["commit"] is False
            assert mock_prune.call_args.kwargs["commit"] is False
            conn.commit.assert_called_once()
            conn.rollback.assert_not_called()

    def test_rolls_back_publication_on_db_error(self):
        from taxcite import server
        from taxcite.chunk import Chunk
        from taxcite.manifest import Publication

        pub = Publication("p501", "Test Pub")
        chunks = [Chunk(pub_id="p501", ordinal=0, first_page=1, last_page=1, text="c0")]
        conn = MagicMock()

        with (
            patch.dict(server._ingest, {}, clear=True),
            patch("taxcite.manifest.CORPUS", (pub,)),
            patch("taxcite.fetch.fetch_publication", return_value="ignored-path"),
            patch("taxcite.parse.parse_pdf", return_value=[]),
            patch("taxcite.chunk.chunk_pages", return_value=chunks),
            patch("taxcite.embed.embed_texts", return_value=[[0.0] * 1024]),
            patch("taxcite.db.get_connection", return_value=conn),
            patch("taxcite.db.release_connection"),
            patch("taxcite.db.upsert_chunk"),
            patch("taxcite.db.prune_chunks", side_effect=psycopg2.OperationalError("connection reset")),
        ):
            server._run_ingest_thread()

            assert server._ingest["state"] == "error"
            conn.rollback.assert_called_once()
            conn.commit.assert_not_called()

    def test_unexpected_bug_class_propagates_instead_of_being_absorbed(self):
        """A defect outside the narrowed except (e.g. an AttributeError from a
        real code bug) must crash instead of being silently reported as an
        ordinary ingest failure."""
        from taxcite import server

        with (
            patch.dict(server._ingest, {}, clear=True),
            patch("taxcite.db.get_connection", return_value=MagicMock()),
            patch("taxcite.db.release_connection"),
            patch("taxcite.fetch.fetch_publication", side_effect=AttributeError("real bug")),
        ):
            with pytest.raises(AttributeError):
                server._run_ingest_thread()


class TestAskResume:
    def test_resume_returns_final_answer(self):
        mock = _mock_graph("Deductible under Pub 936.", [])
        # First get_state call: graph is interrupted; second: completed.
        completed_snapshot = MagicMock()
        completed_snapshot.next = ()
        completed_snapshot.values = {**_BASE_STATE, "answer": "Deductible under Pub 936.", "citations": []}

        interrupted_snapshot = MagicMock()
        interrupted_snapshot.next = ("human_review",)
        interrupted_snapshot.values = _BASE_STATE

        mock.get_state.side_effect = [interrupted_snapshot, completed_snapshot]

        with patch("taxcite.server._graph", mock):
            client = TestClient(app)
            resp = client.post(
                "/ask/resume",
                json={"thread_id": _VALID_THREAD_ID, "approved": True},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "complete"
        assert body["answer"] == "Deductible under Pub 936."

    def test_resume_unknown_thread_id_returns_404(self):
        mock = MagicMock()
        not_found_snapshot = MagicMock()
        not_found_snapshot.next = ()  # no interrupted run
        mock.get_state.return_value = not_found_snapshot

        with patch("taxcite.server._graph", mock):
            client = TestClient(app)
            resp = client.post(
                "/ask/resume",
                json={"thread_id": "11111111-2222-3333-4444-555555555555", "approved": True},
            )

        assert resp.status_code == 404

    def test_resume_non_uuid_thread_id_returns_422(self):
        client = TestClient(app)
        resp = client.post("/ask/resume", json={"thread_id": "not-a-uuid", "approved": True})
        assert resp.status_code == 422

    def test_resume_cost_budget_exceeded_returns_503(self):
        from taxcite.cost import CostBudgetExceeded

        interrupted = MagicMock()
        interrupted.next = ("human_review",)
        interrupted.values = _BASE_STATE
        mock = MagicMock()
        mock.get_state.return_value = interrupted
        mock.invoke.side_effect = CostBudgetExceeded("monthly-budget tripped")

        with patch("taxcite.server._graph", mock):
            client = TestClient(app)
            resp = client.post("/ask/resume", json={"thread_id": _VALID_THREAD_ID, "approved": True})

        assert resp.status_code == 503
        assert "usage limits" in resp.json()["detail"]

    def test_resume_db_operational_error_returns_503(self):
        import psycopg2

        interrupted = MagicMock()
        interrupted.next = ("human_review",)
        interrupted.values = _BASE_STATE
        mock = MagicMock()
        mock.get_state.return_value = interrupted
        mock.invoke.side_effect = psycopg2.OperationalError("connection refused")

        with patch("taxcite.server._graph", mock):
            client = TestClient(app)
            resp = client.post("/ask/resume", json={"thread_id": _VALID_THREAD_ID, "approved": True})

        assert resp.status_code == 503
        assert "Database unavailable" in resp.json()["detail"]

    def test_resume_runtime_error_returns_500(self):
        interrupted = MagicMock()
        interrupted.next = ("human_review",)
        interrupted.values = _BASE_STATE
        mock = MagicMock()
        mock.get_state.return_value = interrupted
        mock.invoke.side_effect = RuntimeError("internal graph error")

        with patch("taxcite.server._graph", mock):
            client = TestClient(app)
            resp = client.post("/ask/resume", json={"thread_id": _VALID_THREAD_ID, "approved": True})

        assert resp.status_code == 500


class TestRateLimit:
    def test_ask_returns_429_when_rate_limit_exhausted(self):
        from slowapi import Limiter
        from slowapi.util import get_remote_address

        original = app.state.limiter
        app.state.limiter = Limiter(key_func=get_remote_address)
        mock = _mock_graph("answer", [])
        try:
            with patch("taxcite.server._graph", mock):
                client = TestClient(app)
                for _ in range(10):  # exhaust the 10/minute limit
                    client.post("/ask", json={"question": "test"})
                resp = client.post("/ask", json={"question": "test"})
        finally:
            app.state.limiter = original

        assert resp.status_code == 429

    def test_resume_returns_429_when_rate_limit_exhausted(self):
        from slowapi import Limiter
        from slowapi.util import get_remote_address

        original = app.state.limiter
        app.state.limiter = Limiter(key_func=get_remote_address)
        not_found = MagicMock()
        not_found.next = ()
        mock = MagicMock()
        mock.get_state.return_value = not_found
        try:
            with patch("taxcite.server._graph", mock):
                client = TestClient(app)
                for _ in range(20):  # exhaust the 20/minute limit
                    client.post("/ask/resume", json={"thread_id": _VALID_THREAD_ID, "approved": True})
                resp = client.post("/ask/resume", json={"thread_id": _VALID_THREAD_ID, "approved": True})
        finally:
            app.state.limiter = original

        assert resp.status_code == 429
