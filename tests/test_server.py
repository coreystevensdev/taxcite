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


def _mock_graph(answer: str = "", citations: list | None = None) -> MagicMock:
    mg = MagicMock()
    mg.invoke.return_value = {**_BASE_STATE, "answer": answer, "citations": citations or []}
    return mg


class TestHealth:
    def test_returns_200(self):
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestAsk:
    def test_returns_answer_and_citations(self):
        citations = [{"pub_id": "p17", "first_page": 3, "last_page": 4}]
        mock = _mock_graph("Yes, it is deductible.", citations)
        with patch("taxcite.server._graph", mock):
            client = TestClient(app)
            resp = client.post("/ask", json={"question": "Is mortgage interest deductible?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Yes, it is deductible."
        assert body["citations"] == citations

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
