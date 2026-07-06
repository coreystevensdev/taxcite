# TaxCite

![CI](https://github.com/coreystevensdev/taxcite/actions/workflows/tests.yml/badge.svg)
![38 tests](https://img.shields.io/badge/tests-38-brightgreen)

[github.com/coreystevensdev/taxcite](https://github.com/coreystevensdev/taxcite)

Agentic RAG system that answers U.S. tax questions with page-level citations from IRS publications. 38 tests (pytest). Ragas eval harness for faithfulness, answer relevancy, and context precision.

## Problem

IRS publications contain the authoritative answers to most individual tax questions, but they are long, cross-referenced PDFs spread across dozens of documents. Searching them returns pages, not answers. A taxpayer asking "can I deduct mortgage interest on a second home?" has to read Pub 936 end to end to find out.

## Solution

A retrieval-augmented generation pipeline that ingests 14 IRS publications into a pgvector index, embeds queries with Voyage AI, retrieves the most relevant passages, and asks Claude to answer with inline citations. Every answer names the publication and page range it came from, so users can verify the source directly.

## Architecture

```
POST /ask
     |
     v
[retrieve]  embed query (voyage-3.5-lite) + pgvector cosine search, top-8 chunks
     |
     +-- empty? --> "No relevant excerpts found"
     |
     v
[human_review]  interrupt()  <-- HITL checkpoint
     |                            graph pauses; client receives thread_id + chunks_preview
     |                            client calls POST /ask/resume {thread_id, approved}
     +-- rejected? --> "Review cancelled"
     |
     v
[generate_answer]  Claude claude-sonnet-4-6, forced submit_answer tool call
     |
     v
{status: "complete", answer, citations: [{pub_id, first_page, last_page}]}
```

The LangGraph state machine has four nodes with two conditional edges. `retrieve` routes to `human_review` when chunks are found, or `no_documents` when the corpus has nothing. `human_review` calls `interrupt()` to pause the graph for human approval of the retrieved excerpts; the graph saves its checkpoint to `MemorySaver`, the server returns an intermediate response with the chunks preview, and `POST /ask/resume` resumes from the saved checkpoint with the human's decision. `generate_answer` forces a structured tool call so citations are always machine-readable rather than extracted from prose.

## Eval Harness

Ragas evaluation over 5 questions across all 14 ingested IRS publications, scoring faithfulness, answer relevancy, and context precision. Scoring uses Ragas with OpenAI as the judge LLM (`OPENAI_API_KEY`); the agent itself uses Anthropic + Voyage AI.

Run after ingestion:

```bash
python -m taxcite eval --dataset eval/dataset.jsonl --out eval/report.json
cat eval/report.json
```

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Retrieval | pgvector (ivfflat cosine) | Native PostgreSQL extension; no separate vector DB service |
| Embeddings | Voyage AI voyage-3.5-lite, 1024-dim | Outperforms OpenAI text-embedding-3-small on retrieval benchmarks at lower cost |
| Agent | LangGraph StateGraph | Explicit state machine separates retrieval and generation; conditional routing is auditable |
| Generation | Anthropic claude-sonnet-4-6 | Forced tool call enforces structured citation output |
| Evaluation | Ragas faithfulness + answer_relevancy + context_precision | Standard RAG eval metrics; reproducible with `python -m taxcite eval` |
| Observability | LangSmith traces via `@traceable` + LangGraph auto-instrumentation | Token costs, latency, state transitions, and raw Anthropic messages in one trace tree |
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

## HITL Interrupt/Resume

The API supports human-in-the-loop review before the generation call. When enabled (always on by default), `POST /ask` may return an intermediate response:

```json
{
  "status": "awaiting_review",
  "thread_id": "uuid",
  "chunks_preview": [
    {"pub_id": "p936", "pages": "pp.12-14", "excerpt": "Mortgage interest on your main home..."}
  ]
}
```

Call `POST /ask/resume` to continue:

```bash
curl -X POST http://localhost:8000/ask/resume \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "<uuid>", "approved": true}'
```

The graph resumes from the `MemorySaver` checkpoint and completes generation. Pass `"approved": false` to cancel without a Claude call.

## LangSmith Tracing

Every `graph.invoke()` call is traced to LangSmith when the following env vars are set:

```bash
export LANGCHAIN_API_KEY=lsv2_pt_...   # from smith.langchain.com
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=taxcite
```

The trace tree per request:

```
LangGraph run
  ├─ retrieve              (node span, auto-instrumented by LangGraph)
  │    └─ embed_query      (embedding span — Voyage AI latency + model)
  ├─ human_review          (node span — shows interrupt payload)
  └─ generate_answer       (llm span — Anthropic messages, token counts, cost)
```

Traces show: LangGraph state transitions (retrieve -> human_review -> generate_answer), the HITL interrupt point and resume event, the raw Anthropic messages payload, token counts and cost per node, Voyage AI embedding latency, and end-to-end latency across both the initial invoke and the resume call.

## Known Limitations

- Context window: retrieves top-8 chunks per question; multi-part questions spanning many publications may miss relevant context.
- No OCR: `pdfplumber` extracts digital text only; scanned pages (some older IRS pubs) are silently skipped.
- Per-instance state: both the rate limiter and the HITL `MemorySaver` checkpointer are in-process. Interrupted threads are lost on restart and not shared across replicas; replace `MemorySaver` with `PostgresSaver` for production durability.
- Ragas judge uses OpenAI by default: evaluation cost is separate from inference cost and requires an additional API key.
- Publications are ingested as static snapshots; re-ingest when IRS revises a publication (annual cycle for most).

## License

MIT
