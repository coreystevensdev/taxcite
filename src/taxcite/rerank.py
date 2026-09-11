"""Cross-encoder reranking of retrieved chunks via voyage rerank-2.

Vector search puts the query and the chunk into the same space independently, so
it scores what a chunk is broadly about rather than whether it answers this
question. A reranker reads the pair together and can tell a passage that mentions
mortgage interest from one that answers a question about deducting it.

Retrieval therefore pulls a wider candidate set than the model will see and this
narrows it, which is where the precision comes from: the shortlist is chosen from
24 rather than being whatever the first 8 nearest vectors were.
"""

from __future__ import annotations

import os

import voyageai
from langsmith import traceable

from taxcite import cost

RERANK_MODEL = "rerank-2"

_client: voyageai.Client | None = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _client


def _guard(total_tokens: int) -> None:
    decision = cost.rerank_cap.evaluate(total_tokens * cost.VOYAGE_RERANK_COST_PER_TOKEN)
    if not decision.allowed:
        raise cost.CostBudgetExceeded(
            f"rerank blocked by cost cap ({decision.trip}): "
            f"${decision.observed:.6f} observed, ${decision.monthly_spend:.4f} monthly"
        )


@traceable(name="rerank", run_type="retriever")
def rerank(query: str, chunks: list, top_k: int) -> list:
    """Return the top_k chunks by relevance to query, most relevant first.

    A candidate set already at or under top_k is returned untouched: there is
    nothing to select and paying for a reorder of everything the model will read
    anyway buys nothing.
    """
    if not chunks or len(chunks) <= top_k:
        return chunks

    response = _get_client().rerank(
        query=query,
        documents=[c.text for c in chunks],
        model=RERANK_MODEL,
        top_k=top_k,
    )
    _guard(response.total_tokens)
    return [chunks[r.index] for r in response.results]
