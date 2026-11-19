"""Whether the page can offer a live answer, and what it falls back to.

The ask box has to work before the keys land or the product half of the demo
is a dead control. It answers from the 50 recorded eval runs in that case, and
the mode has to reach the page, because a recorded answer presented as a live
one is a claim the deployment has not earned.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPORT = Path(__file__).resolve().parent.parent / "eval" / "report.json"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    from taxcite.server import app

    return TestClient(app)


def test_status_says_answering_is_not_configured(client):
    body = client.get("/status").json()

    assert body["answering"] == "not_configured"


def test_status_says_live_once_both_keys_are_set(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")

    assert client.get("/status").json()["answering"] == "live"


def test_one_key_alone_is_not_live(client, monkeypatch):
    """Both are needed to answer, so one is the same as none."""
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")

    assert client.get("/status").json()["answering"] == "not_configured"


def test_the_recorded_answers_carry_citations(client):
    """What the fallback serves. An answer without the publication it came from
    is not what this product claims to do."""
    report = json.loads(REPORT.read_text())

    assert len(report["per_question"]) == 50
    for q in report["per_question"]:
        assert q["user_input"].strip()
        assert q["response"].strip()
        assert q["retrieved_contexts"], "a recorded answer with no passages cites nothing"


def test_status_does_not_leak_the_keys(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-secret-value")

    body = client.get("/status").text

    assert "secret-value" not in body


def test_status_survives_a_database_with_no_schema_yet(client, monkeypatch):
    """Before the first ingest the chunks table does not exist. /status is the
    endpoint you check to find that out, so it cannot be the one that crashes:
    it reported UndefinedTable as a 500 against a fresh database."""
    import psycopg2

    from taxcite import db

    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@localhost/nothing")
    monkeypatch.setattr(db, "get_connection", lambda: object())
    monkeypatch.setattr(db, "release_connection", lambda c: None)

    def no_table(_conn):
        raise psycopg2.errors.UndefinedTable('relation "chunks" does not exist')

    monkeypatch.setattr(db, "count_chunks", no_table)

    resp = client.get("/status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks_by_pub"] == {}
    assert body["answering"] == "not_configured"
