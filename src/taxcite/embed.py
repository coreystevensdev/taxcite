"""Embedding via voyage-3.5-lite (1024-dim)."""
from __future__ import annotations

import os

import voyageai

EMBED_MODEL = "voyage-3.5-lite"
EMBED_DIMS = 1024
BATCH_SIZE = 128

_client: voyageai.Client | None = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document strings in batches of 128. Returns one 1024-dim vector per text."""
    client = _get_client()
    result: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embed(batch, model=EMBED_MODEL, input_type="document")
        result.extend(response.embeddings)
    return result


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    client = _get_client()
    response = client.embed([text], model=EMBED_MODEL, input_type="query")
    return response.embeddings[0]
