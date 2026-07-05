from __future__ import annotations

from unittest.mock import MagicMock, patch


FAKE_EMBEDDING = [[0.1] * 1024]


def _make_response(embeddings):
    r = MagicMock()
    r.embeddings = embeddings
    return r


@patch("taxcite.embed._get_client")
def test_embed_texts_returns_one_vector_per_text(mock_get_client):
    client = MagicMock()
    client.embed.return_value = _make_response([[0.1] * 1024, [0.2] * 1024])
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
        _make_response([[0.1] * 1024] * 128),
        _make_response([[0.2] * 1024] * 10),
    ]
    mock_get_client.return_value = client

    from taxcite.embed import embed_texts

    result = embed_texts(["t"] * 138)
    assert len(result) == 138
    assert client.embed.call_count == 2


@patch("taxcite.embed._get_client")
def test_embed_query_uses_query_input_type(mock_get_client):
    client = MagicMock()
    client.embed.return_value = _make_response([[0.3] * 1024])
    mock_get_client.return_value = client

    from taxcite.embed import embed_query

    result = embed_query("what is the standard deduction?")
    assert len(result) == 1024
    client.embed.assert_called_once_with(
        ["what is the standard deduction?"], model="voyage-3.5-lite", input_type="query"
    )
