"""Tests for the FastAPI server endpoints."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from taxcite.server import app

_BASE_STATE = {
    "question": "q",
    "chunks": [],
    "answer": "",
    "citations": [],
}


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
            assert body["error"] == "RuntimeError: boom"


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
                json={"thread_id": "abc-123", "approved": True},
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
                json={"thread_id": "does-not-exist", "approved": True},
            )

        assert resp.status_code == 404
