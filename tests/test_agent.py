"""Unit tests for the LangGraph agent nodes and routing logic."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from taxcite.agent import (
    AgentState,
    _route_after_retrieve,
    _route_after_review,
    generate_answer,
    human_review,
    no_documents,
    rejected,
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


# retrieve node

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


# routing after retrieve

def test_route_returns_human_review_when_chunks_present():
    assert _route_after_retrieve(_state_with_chunks()) == "human_review"


def test_route_returns_no_documents_when_chunks_empty():
    assert _route_after_retrieve(_EMPTY_STATE) == "no_documents"


# routing after human_review

def test_route_after_review_approved():
    state = {**_state_with_chunks(), "human_approved": True}
    assert _route_after_review(state) == "generate_answer"


def test_route_after_review_rejected():
    state = {**_state_with_chunks(), "human_approved": False}
    assert _route_after_review(state) == "rejected"


def test_route_after_review_none_treated_as_rejected():
    # human_approved absent or None routes to rejected
    assert _route_after_review(_state_with_chunks()) == "rejected"


# human_review node

def test_human_review_approved_sets_state():
    with patch("taxcite.agent.interrupt", return_value=True):
        result = human_review(_state_with_chunks())
    assert result == {"human_approved": True}


def test_human_review_rejected_sets_state():
    with patch("taxcite.agent.interrupt", return_value=False):
        result = human_review(_state_with_chunks())
    assert result == {"human_approved": False}


def test_human_review_passes_chunks_preview_to_interrupt():
    with patch("taxcite.agent.interrupt", return_value=True) as mock_interrupt:
        human_review(_state_with_chunks())

    call_args = mock_interrupt.call_args[0][0]
    assert "chunks_preview" in call_args
    assert call_args["chunks_preview"][0]["pub_id"] == "p17"


# generate_answer node

def _make_tool_response(answer: str, citations: list[dict]) -> MagicMock:
    tool_block = SimpleNamespace(
        type="tool_use",
        name="submit_answer",
        input={"answer": answer, "citations": citations},
    )
    response = MagicMock()
    response.content = [tool_block]
    response.usage = SimpleNamespace(input_tokens=200, output_tokens=100)
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


def test_generate_answer_raises_cost_budget_exceeded_when_cap_trips():
    from taxcite import cost
    from taxcite.cost import CostBudgetExceeded

    mock_response = _make_tool_response("answer", [])

    tripped = MagicMock()
    tripped.allowed = False
    tripped.trip = "monthly-budget"
    tripped.observed = 0.005
    tripped.monthly_spend = 10.01

    with (
        patch("taxcite.agent.anthropic.Anthropic") as MockClient,
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        patch.object(cost.cap, "evaluate", return_value=tripped),
    ):
        MockClient.return_value.messages.create.return_value = mock_response
        with pytest.raises(CostBudgetExceeded, match="monthly-budget"):
            generate_answer(_state_with_chunks())


# no_documents node

def test_no_documents_returns_fallback_with_empty_citations():
    result = no_documents(_EMPTY_STATE)

    assert "No relevant" in result["answer"]
    assert result["citations"] == []


# rejected node

def test_rejected_returns_cancellation_message():
    result = rejected(_EMPTY_STATE)

    assert "cancelled" in result["answer"].lower()
    assert result["citations"] == []
