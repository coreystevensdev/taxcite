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

    with patch.object(cost.cap, "evaluate", return_value=tripped):
        from taxcite.embed import embed_query

        with pytest.raises(CostBudgetExceeded, match="absolute-ceiling"):
            embed_query("test")
