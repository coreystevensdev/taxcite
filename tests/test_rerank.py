"""Reranking narrows a candidate set to the chunks the model actually reads."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from taxcite.rerank import rerank


@dataclass
class _Chunk:
    text: str


def _voyage_response(indices, total_tokens=500):
    resp = MagicMock()
    resp.results = [MagicMock(index=i) for i in indices]
    resp.total_tokens = total_tokens
    return resp


def test_reorders_and_truncates_to_top_k():
    chunks = [_Chunk(f"chunk {i}") for i in range(5)]
    with patch("taxcite.rerank._get_client") as client:
        client.return_value.rerank.return_value = _voyage_response([3, 0])
        out = rerank("q", chunks, top_k=2)
    assert [c.text for c in out] == ["chunk 3", "chunk 0"]


def test_skips_the_call_when_there_is_nothing_to_choose_between():
    """Paying to reorder a set the model will read in full buys nothing."""
    chunks = [_Chunk("a"), _Chunk("b")]
    with patch("taxcite.rerank._get_client") as client:
        out = rerank("q", chunks, top_k=8)
    client.assert_not_called()
    assert out == chunks


def test_empty_candidates_pass_straight_through():
    with patch("taxcite.rerank._get_client") as client:
        assert rerank("q", [], top_k=8) == []
    client.assert_not_called()


def test_cost_cap_blocks_and_names_the_rerank():
    from taxcite import cost

    chunks = [_Chunk(f"chunk {i}") for i in range(5)]
    tripped = MagicMock(allowed=False, trip="rolling-median", observed=0.9, monthly_spend=1.2)
    with (
        patch("taxcite.rerank._get_client") as client,
        patch.object(cost.rerank_cap, "evaluate", return_value=tripped),
    ):
        client.return_value.rerank.return_value = _voyage_response([0, 1])
        try:
            rerank("q", chunks, top_k=2)
        except cost.CostBudgetExceeded as exc:
            assert "rerank blocked by cost cap" in str(exc)
        else:
            raise AssertionError("expected CostBudgetExceeded")
