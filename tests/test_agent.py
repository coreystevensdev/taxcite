"""Unit tests for the LangGraph agent nodes and routing logic."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from taxcite.agent import (
    AgentState,
    _route_after_retrieve,
    generate_answer,
    no_documents,
    retrieve,
)
from taxcite.chunk import Chunk

_FAKE_VEC = [0.0] * 1024
_FAKE_CHUNK = Chunk(pub_id="p17", ordinal=0, first_page=3, last_page=4, text="Interest is deductible.")

_EMPTY_STATE: AgentState = {
    "question": "Can I deduct mortgage interest?",
    "chunks": [],
    "answer": "",
    "citations": [],
}


def _state_with_chunks() -> AgentState:
    return {**_EMPTY_STATE, "chunks": [_FAKE_CHUNK]}


# ---- retrieve node ----

def test_retrieve_embeds_question_and_returns_chunks():
    mock_conn = MagicMock()
    with (
        patch("taxcite.agent.embed.embed_query", return_value=_FAKE_VEC) as mock_embed,
        patch("taxcite.agent.db.get_connection", return_value=mock_conn),
        patch("taxcite.agent.db.search_chunks", return_value=[_FAKE_CHUNK]) as mock_search,
    ):
        result = retrieve(_EMPTY_STATE)

    mock_embed.assert_called_once_with("Can I deduct mortgage interest?")
    mock_search.assert_called_once_with(mock_conn, _FAKE_VEC, top_k=8)
    assert result["chunks"] == [_FAKE_CHUNK]
    mock_conn.close.assert_called_once()


def test_retrieve_closes_connection_on_search_error():
    mock_conn = MagicMock()
    with (
        patch("taxcite.agent.embed.embed_query", return_value=_FAKE_VEC),
        patch("taxcite.agent.db.get_connection", return_value=mock_conn),
        patch("taxcite.agent.db.search_chunks", side_effect=RuntimeError("db down")),
    ):
        with pytest.raises(RuntimeError):
            retrieve(_EMPTY_STATE)

    mock_conn.close.assert_called_once()


# ---- routing ----

def test_route_returns_generate_when_chunks_present():
    assert _route_after_retrieve(_state_with_chunks()) == "generate_answer"


def test_route_returns_no_documents_when_chunks_empty():
    assert _route_after_retrieve(_EMPTY_STATE) == "no_documents"


# ---- generate_answer node ----

def _make_tool_response(answer: str, citations: list[dict]) -> MagicMock:
    tool_block = SimpleNamespace(
        type="tool_use",
        name="submit_answer",
        input={"answer": answer, "citations": citations},
    )
    response = MagicMock()
    response.content = [tool_block]
    return response


def test_generate_answer_parses_tool_call():
    expected_answer = "Yes, mortgage interest is deductible [p17, pp.3-4]."
    expected_citations = [{"pub_id": "p17", "first_page": 3, "last_page": 4}]
    mock_response = _make_tool_response(expected_answer, expected_citations)

    with (
        patch("taxcite.agent.anthropic.Anthropic") as MockClient,
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
    ):
        MockClient.return_value.messages.create.return_value = mock_response
        result = generate_answer(_state_with_chunks())

    assert result["answer"] == expected_answer
    assert result["citations"] == expected_citations


def test_generate_answer_formats_multi_page_citation_correctly():
    """Verifies pp.X-Y format for multi-page chunks vs p.X for single-page."""
    two_page_chunk = Chunk(pub_id="p936", ordinal=1, first_page=5, last_page=6, text="text")
    state = {**_EMPTY_STATE, "chunks": [two_page_chunk]}

    mock_response = _make_tool_response("answer", [])
    with (
        patch("taxcite.agent.anthropic.Anthropic") as MockClient,
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
    ):
        mock_create = MockClient.return_value.messages.create
        mock_create.return_value = mock_response
        generate_answer(state)
        call_kwargs = mock_create.call_args
        user_content = call_kwargs[1]["messages"][0]["content"]

    assert "pp.5-6" in user_content


# ---- no_documents node ----

def test_no_documents_returns_fallback_with_empty_citations():
    result = no_documents(_EMPTY_STATE)

    assert "No relevant" in result["answer"]
    assert result["citations"] == []
