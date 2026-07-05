# TaxCite

Agentic RAG system that answers U.S. tax questions with page-level citations from IRS publications. 28 tests (pytest). Ragas eval harness for faithfulness, answer relevancy, and context precision.

## Problem

IRS publications contain the authoritative answers to most individual tax questions, but they are long, cross-referenced PDFs spread across dozens of documents. Searching them returns pages, not answers. A taxpayer asking "can I deduct mortgage interest on a second home?" has to read Pub 936 end to end to find out.

## Solution

A retrieval-augmented generation pipeline that ingests 14 IRS publications into a pgvector index, embeds queries with Voyage AI, retrieves the most relevant passages, and asks Claude to answer with inline citations. Every answer names the publication and page range it came from, so users can verify the source directly.

## Architecture

```
User question
     |
     v
[embed_query]  (voyage-3.5-lite, 1024-dim, query input_type)
     |
     v
[pgvector cosine search]  top-8 chunks from 14 IRS pubs
     |
     +-- empty? --> "No relevant excerpts found"
     |
     v
[Claude claude-sonnet-4-6]  system: "answer from excerpts only, cite pages"
  forced tool call: submit_answer(answer, citations)
     |
     v
{answer, citations: [{pub_id, first_page, last_page}]}
```

The LangGraph state machine separates retrieval from generation: the `retrieve` node embeds and searches, a conditional edge routes to `no_documents` when the corpus has nothing, and `generate_answer` forces a structured tool call so citations are always machine-readable rather than extracted from prose.

## Eval Scores

Ragas evaluation over 5 questions from the eval dataset after ingesting all 14 publications.

| Metric | Score |
|---|---|
| Faithfulness | TBD (run `python -m taxcite eval` after ingestion) |
| Answer Relevancy | TBD |
| Context Precision | TBD |

Scoring uses Ragas with OpenAI as the judge LLM (`OPENAI_API_KEY`). The agent itself uses Anthropic + Voyage AI.

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Retrieval | pgvector (ivfflat cosine) | Native PostgreSQL extension; no separate vector DB service |
| Embeddings | Voyage AI voyage-3.5-lite, 1024-dim | Outperforms OpenAI text-embedding-3-small on retrieval benchmarks at lower cost |
| Agent | LangGraph StateGraph | Explicit state machine separates retrieval and generation; conditional routing is auditable |
| Generation | Anthropic claude-sonnet-4-6 | Forced tool call enforces structured citation output |
| Evaluation | Ragas faithfulness + answer_relevancy + context_precision | Standard RAG eval metrics; reproducible with `python -m taxcite eval` |
| API | FastAPI + uvicorn | Typed request/response schemas; easy local testing with TestClient |
| PDF parsing | pdfplumber | Page-accurate text extraction with page-number tracking for citations |
| Chunking | Custom overlap chunker | 1600-char target, one-paragraph overlap, page range tracking per chunk |

## Getting Started

```bash
docker compose up -d db        # start pgvector
cp .env.example .env           # fill in API keys
pip install -e ".[eval]"
python -m taxcite ingest       # fetch + parse + chunk + embed all 14 pubs (~5-10 min)
python -m taxcite serve        # start API at http://localhost:8000
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I deduct mortgage interest on my primary home?"}'
```

Or with Docker Compose:

```bash
cp .env.example .env
docker compose up -d
docker compose exec api python -m taxcite ingest
```

## Running the Eval

```bash
python -m taxcite eval --dataset eval/dataset.jsonl --out eval/report.json
cat eval/report.json
```

Requires `OPENAI_API_KEY` for the Ragas judge LLM.

## Known Limitations

- Context window: retrieves top-8 chunks per question; multi-part questions spanning many publications may miss relevant context.
- No OCR: `pdfplumber` extracts digital text only; scanned pages (some older IRS pubs) are silently skipped.
- Per-instance rate limiting: the in-memory rate limiter is not shared across API replicas; horizontal scaling requires a shared store.
- Ragas judge uses OpenAI by default: evaluation cost is separate from inference cost and requires an additional API key.
- Publications are ingested as static snapshots; re-ingest when IRS revises a publication (annual cycle for most).

## License

MIT
