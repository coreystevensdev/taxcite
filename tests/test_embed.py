from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

FAKE_EMBEDDING = [[0.1] * 1024]


def _make_response(embeddings, total_tokens: int = 50):
    r = MagicMock()
    r.embeddings = embeddings
    r.total_tokens = total_tokens
    return r


@patch("taxcite.embed._get_client")
def test_embed_texts_returns_one_vector_per_text(mock_get_client):
    client = MagicMock()
    client.embed.return_value = _make_response([[0.1] * 1024, [0.2] * 1024], total_tokens=100)
    mock_get_client.return_value = client

    from taxcite.embed import embed_texts

    result = embed_texts(["text one", "text two"])
    assert len(result) == 2
    assert len(result[0]) == 1024
    client.embed.assert_called_once_with(
        ["text one", "text two"], model="voyage-3.5-lite", input_type="document"
    )


@patch("taxcite.embed._get_client")
def test_embed_texts_batches_at_128(mock_get_client):
    client = MagicMock()
    client.embed.side_effect = [
        _make_response([[0.1] * 1024] * 128, total_tokens=6400),
        _make_response([[0.2] * 1024] * 10, total_tokens=500),
    ]
    mock_get_client.return_value = client

    from taxcite.embed import embed_texts

    result = embed_texts(["t"] * 138)
    assert len(result) == 138
    assert client.embed.call_count == 2


@patch("taxcite.embed._get_client")
def test_embed_query_uses_query_input_type(mock_get_client):
    client = MagicMock()
    client.embed.return_value = _make_response([[0.3] * 1024], total_tokens=10)
    mock_get_client.return_value = client

    from taxcite.embed import embed_query

    result = embed_query("what is the standard deduction?")
    assert len(result) == 1024
    client.embed.assert_called_once_with(
        ["what is the standard deduction?"], model="voyage-3.5-lite", input_type="query"
    )


@patch("taxcite.embed._get_client")
def test_embed_query_raises_embedding_error_on_empty_response(mock_get_client):
    from taxcite.embed import EmbeddingError, embed_query

    client = MagicMock()
    client.embed.return_value = _make_response([], total_tokens=10)
    mock_get_client.return_value = client

    with pytest.raises(EmbeddingError, match="no embeddings"):
        embed_query("what is the standard deduction?")


@patch("taxcite.embed._get_client")
def test_embed_query_raises_cost_budget_exceeded_when_cap_trips(mock_get_client):
    from taxcite import cost
    from taxcite.cost import CostBudgetExceeded

    client = MagicMock()
    client.embed.return_value = _make_response([[0.1] * 1024], total_tokens=50)
    mock_get_client.return_value = client

    tripped = MagicMock()
    tripped.allowed = False
    tripped.trip = "absolute-ceiling"
    tripped.observed = 0.20
    tripped.monthly_spend = 0.20

    # query_embed_cap, not embed_cap: embed_query stopped sharing the ingest cap
    # because a batch of 128 chunks and one question have costs three orders of
    # magnitude apart, so the shared rolling median described neither and killed
    # the public demo on ordinary question length.
    with patch.object(cost.query_embed_cap, "evaluate", return_value=tripped):
        from taxcite.embed import embed_query

        with pytest.raises(CostBudgetExceeded, match="absolute-ceiling"):
            embed_query("test")


@patch("taxcite.embed._get_client")
def test_embed_query_does_not_consult_the_ingest_cap(mock_get_client):
    """The regression that killed the demo: a question long enough to exceed 3x
    the median of recent short questions returned 503 after $0.0000024 of spend.
    embed_query must not be judged against the window embed_texts fills."""
    from taxcite import cost

    client = MagicMock()
    client.embed.return_value = _make_response([[0.1] * 1024], total_tokens=34)
    mock_get_client.return_value = client

    from taxcite.embed import embed_query

    with patch.object(cost.embed_cap, "evaluate") as ingest_cap:
        embed_query("a question long enough to be judged an anomaly by the old cap")

    ingest_cap.assert_not_called()


def test_query_cap_allows_a_question_several_times_the_running_median():
    """Six short questions then a long one. Under the shared 3x cap this was the
    exact sequence that returned 503; the query cap has to pass it."""
    from taxcite import cost

    per_token = cost.VOYAGE_COST_PER_TOKEN
    for tokens in (8, 7, 9, 6, 8, 7):
        assert cost.query_embed_cap.evaluate(tokens * per_token).allowed

    # 34 tokens is about 4.5x the median of the six above
    decision = cost.query_embed_cap.evaluate(34 * per_token)
    assert decision.allowed, f"blocked via {decision.trip}"
