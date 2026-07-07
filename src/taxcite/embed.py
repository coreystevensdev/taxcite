"""Embedding via voyage-3.5-lite (1024-dim)."""
from __future__ import annotations

import os

import voyageai
from langsmith import traceable

from taxcite import cost

EMBED_MODEL = "voyage-3.5-lite"
EMBED_DIMS = 1024
BATCH_SIZE = 128

_client: voyageai.Client | None = None


class EmbeddingError(RuntimeError):
    pass


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _client


def _guard(total_tokens: int) -> None:
    decision = cost.cap.evaluate(total_tokens * cost.VOYAGE_COST_PER_TOKEN)
    if not decision.allowed:
        raise cost.CostBudgetExceeded(
            f"embedding blocked by cost cap ({decision.trip}): "
            f"${decision.observed:.6f} observed, ${decision.monthly_spend:.4f} monthly"
        )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document strings in batches of 128. Returns one 1024-dim vector per text."""
    client = _get_client()
    result: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embed(batch, model=EMBED_MODEL, input_type="document")
        _guard(response.total_tokens)
        result.extend(response.embeddings)
    return result


@traceable(name="embed_query", run_type="embedding")
def embed_query(text: str) -> list[float]:
    """input_type="query" tells Voyage to optimize the vector for search rather than storage.
    Use embed_texts() (input_type="document") when ingesting corpus chunks.
    """
    client = _get_client()
    response = client.embed([text], model=EMBED_MODEL, input_type="query")
    _guard(response.total_tokens)
    if not response.embeddings:
        raise EmbeddingError("Voyage returned no embeddings for the query")
    return response.embeddings[0]
