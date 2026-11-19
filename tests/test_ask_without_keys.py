"""What /ask does on a deployment whose vendor keys are not set yet.

Without them the retrieve node raised KeyError deep inside the graph, which
reaches a visitor as a 500 and a stack trace. The endpoint has to say what is
missing instead, because a public URL that answers with a traceback is worse
than one that answers with a sentence.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    from taxcite.server import app

    return TestClient(app)


def test_ask_without_keys_is_a_sentence_not_a_traceback(client):
    resp = client.post("/ask", json={"question": "Can I deduct mortgage interest?"})

    assert resp.status_code == 503
    detail = str(resp.json().get("detail", "")).lower()
    assert "keyerror" not in detail
    assert "traceback" not in detail


def test_the_message_names_what_is_missing(client):
    """So whoever deploys it can fix it without reading the source."""
    detail = str(client.post("/ask", json={"question": "x"}).json()["detail"])

    assert "VOYAGE_API_KEY" in detail or "ANTHROPIC_API_KEY" in detail


def test_one_missing_key_is_enough_to_refuse(client, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "set-but-anthropic-is-not")

    resp = client.post("/ask", json={"question": "x"})

    assert resp.status_code == 503
    assert "ANTHROPIC_API_KEY" in str(resp.json()["detail"])


def test_the_eval_page_is_unaffected_by_missing_keys(client):
    """It reads a committed file, so it owes nothing to either vendor."""
    assert client.get("/").status_code == 200
    assert client.get("/eval/report.json").status_code == 200
