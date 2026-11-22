# TaxCite

![CI](https://github.com/coreystevensdev/taxcite/actions/workflows/tests.yml/badge.svg)
![123 tests](https://img.shields.io/badge/tests-123-brightgreen)

**[Try it: taxcite.coreystevens.dev](https://taxcite.coreystevens.dev/)**: the page opens on the eval, so you see 50 real questions with the answer given, the passages retrieval pulled, and what each one scored, worst first.

## Overview

Ask a US tax question, get an answer that names the IRS publication and page it came from, so you can go read the source yourself rather than trusting the model.

It is a retrieval-augmented generation pipeline over 14 IRS publications, with a human approval step before an answer is generated. Answer quality is measured rather than asserted: a Ragas harness scores faithfulness, answer relevancy and context precision over 50 questions, and the per-question scores are committed alongside the aggregates. 123 tests (pytest).

## Problem

IRS publications contain the authoritative answers to most individual tax questions, but they are long, cross-referenced PDFs spread across dozens of documents. Searching them returns pages, not answers. A taxpayer asking "can I deduct mortgage interest on a second home?" has to read Pub 936 end to end to find out.

## Solution

A retrieval-augmented generation pipeline that ingests 14 IRS publications into a pgvector index, embeds queries with Voyage AI, retrieves the most relevant passages, and asks Claude to answer with inline citations. Every answer names the publication and page range it came from, so users can verify the source directly.

## Architecture

```mermaid
flowchart TD
    A["POST /ask"] --> B["retrieve\nembed query via Voyage AI voyage-3.5-lite\npgvector cosine search, 24 candidates\nrerank-2 cross-encoder narrows to 8"]
    B -->|empty| C["No relevant excerpts found"]
    B -->|chunks found| D["human_review\ninterrupt(): graph pauses\nclient receives thread_id + chunks_preview"]
    D -->|approved: false| E["Review cancelled"]
    D -->|approved: true| F["generate_answer\nClaude claude-sonnet-4-6\nforced submit_answer tool call"]
    F --> G["{status: complete, answer,\ncitations: [{pub_id, first_page, last_page}]}"]
```

The LangGraph state machine has five nodes with two conditional edges. `retrieve` routes to `human_review` when chunks are found, or `no_documents` when the corpus has nothing. `human_review` calls `interrupt()` to pause the graph for human approval of the retrieved excerpts; the graph saves its checkpoint to `MemorySaver`, the server returns an intermediate response with the chunks preview, and `POST /ask/resume` resumes from the saved checkpoint with the human's decision. `generate_answer` forces a structured tool call so citations are always machine-readable rather than extracted from prose.

## Routes

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Liveness check, no dependencies |
| GET | `/status` | Ingest state (`idle`/`running`/`ready`/`error`) plus chunk counts per publication |
| POST | `/ask` | Runs the graph; returns a complete answer or pauses at `human_review` with a chunks preview |
| POST | `/ask/resume` | Resumes a paused thread from its `MemorySaver` checkpoint with the human's approve/reject decision |

## Eval Harness

Ragas evaluation over 50 questions spanning all 14 ingested IRS publications, scoring faithfulness, answer relevancy, and context precision. Questions include numeric exact-match cases (dollar thresholds, age limits, percentages) and conceptual retrieval cases. Claude judges and Voyage embeds, so the eval needs no third provider; the judge's embedder is `voyage-3` while the agent retrieves with `voyage-3.5-lite`, same vendor and same 1024 dimensions but not the same model. Every question's own scores are kept in the report alongside the aggregate, which is what turns a retrieval change into a paired comparison rather than two means that might differ by noise.

The run committed at `eval/report.json`, from 2026-09-11. It predates the `provenance` and `score_coverage` fields the harness writes now, so it carries no model version or code revision and cannot be compared against a later run as a paired measurement. A fresh run produces both:

| Metric | Score |
|---|---|
| Faithfulness | 0.945 |
| Answer relevancy | 0.633 |
| Context precision | 0.857 |

The corpus is pinned to a tax year, so these are measured against the publications the dataset was written against. Before pinning, `ingest` pulled whatever revision the IRS was currently serving: 7 of the 50 questions asked for 2024 figures the 2025 revisions do not contain, the agent correctly declined to state them, and Ragas scored those refusals as total misses. That alone was worth 0.759 against 0.857 on context precision and 0.911 against 0.945 on faithfulness.

Retrieval pulls 24 candidates and a voyage `rerank-2` cross-encoder narrows them to the 8 the model reads. Vector search scores the query and the chunk in the same space independently, so it measures what a passage is about rather than whether it answers this question; a cross-encoder reads the pair together. Measured against the same corpus without it, context precision went 0.704 to 0.759, faithfulness 0.891 to 0.911, and answer relevancy down 0.023. Two reranked runs are committed to size the noise under those gaps, [`eval/runs/main-lines.json`](eval/runs/main-lines.json) and [`eval/runs/branch-sentences.json`](eval/runs/branch-sentences.json), with all 50 per-question scores in each, so the spread this paragraph rests on is a file you can open rather than a number to take on trust. Each metric gets its own spread, because they are not alike: 0.0015 on context precision, 0.0014 on answer relevancy, and 0.0122 on faithfulness, which is eight times noisier than the other two. Against those, the context-precision gain is 37x its spread and the one result here that is actually solid. Faithfulness is 1.6x, which is not an effect, and reading it as one was the error this paragraph used to make. Answer relevancy is 16x its spread and moved the wrong way, so the reranker buys ranking quality and costs something on relevancy rather than being free.

Those numbers are worth reading alongside what they were a day earlier: **0.464 / 0.136 / 0.102**. The harness had never completed a run, so nothing had ever measured the retrieval path, and three bugs were sitting in it. PDF text came out of `pdfplumber` with the two columns zippered together, so every chunk was two half-sentences from unrelated passages. The IVFFlat index was created during migration, before any rows existed, and IVFFlat derives its centroids from the data present at build time, so on an empty table a query probed one meaningless list out of a hundred: 11 of the 50 questions retrieved nothing at all. Generation then ran at `max_tokens=1024`, which was survivable only while retrieval was returning nothing.

Each one hid the next. An eval that has never run is not a weak signal, it is no signal.

Run after ingestion:

```bash
python -m taxcite eval --dataset eval/dataset.jsonl --out eval/report.json
cat eval/report.json
```

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Retrieval | pgvector (HNSW, cosine) | Native PostgreSQL extension; no separate vector DB service. HNSW rather than IVFFlat because the index is built at migration time, before any rows exist |
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
```

Compose needs no ingest step. `server.py` starts one in a background thread on
first boot when the corpus is empty and `VOYAGE_API_KEY` is set, which is the
state a fresh `compose up` with a filled-in `.env` produces. `/health` answers
200 straight away and queries return nothing until it finishes, roughly 5 to 10
minutes. Running `taxcite ingest` as well embeds all 14 publications a second
time, under a separate cost cap, for rows that upsert over the first set.

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
  │    └─ embed_query      (embedding span: Voyage AI latency + model)
  ├─ human_review          (node span: shows interrupt payload)
  └─ generate_answer       (llm span: Anthropic messages, token counts, cost)
```

Traces show: LangGraph state transitions (retrieve -> human_review -> generate_answer), the HITL interrupt point and resume event, the raw Anthropic messages payload, token counts and cost per node, Voyage AI embedding latency, and end-to-end latency across both the initial invoke and the resume call.

## How this was built

AI assistance was used, and the thing worth reporting is that the first version of this pipeline scored 0.464 faithfulness, 0.136 answer relevancy and 0.102 context precision while looking entirely finished. Every component was plausible. Three bugs were sitting in it and each one hid the next.

`pdfplumber` returned the two columns of each IRS page zippered together, so every chunk was two half-sentences from unrelated passages. The IVFFlat index was built during migration before any rows existed, and IVFFlat derives its centroids from the data present at build time, so on an empty table a query probed one meaningless list out of a hundred: 11 of the 50 questions retrieved nothing at all. Generation then ran at `max_tokens=1024`, which was survivable only because retrieval was returning nothing to write about.

None of that was visible from reading the code, and none of it would have been caught by a test that mocked the database or the PDF reader. It took running the eval end to end against real publications. The lesson I took from it is in the section above: an eval that has never completed a run is not a weak signal, it is no signal, and a confident-looking RAG pipeline is exactly the kind of artifact that hides this.

Two guards exist because of it. The judge is wired explicitly in `_judge_llm` rather than left to whatever Ragas reaches for, so scoring cannot silently fall back to a provider this project does not configure. And every question's own score is committed next to the aggregate in `eval/report.json`, which is what makes a retrieval change a paired comparison rather than two means that might differ by noise. Both runs behind the reranker comparison are committed too, so the run-to-run spread the argument rests on is a file rather than a claim.

Answer relevancy at 0.633 is the weakest of the three scores and it stays in Known Limitations rather than being explained away.

## Failure Modes

[docs/failure-modes.md](docs/failure-modes.md) covers how each dependency fails, which tracer can actually see it, and what to do. It is written against the current node names and schema rather than from memory.

## Known Limitations

- Context window: retrieves top-8 chunks per question; multi-part questions spanning many publications may miss relevant context.
- No OCR: `pdfplumber` extracts digital text only; scanned pages (some older IRS pubs) are silently skipped.
- Per-instance state: both the rate limiter and the HITL `MemorySaver` checkpointer are in-process. Interrupted threads are lost on restart and not shared across replicas; replace `MemorySaver` with `PostgresSaver` for production durability.
- **The corpus is pinned to one tax year, and the dataset has to move with it.** `TAX_YEAR` in `manifest.py` selects which revision `ingest` fetches, from `pub/irs-prior` where publications are addressed by year, rather than `pub/irs-pdf` which always serves the current one. Changing it means rewriting the dataset's dollar figures to match; leaving them out of step scores correct refusals as failures, which is what happened before pinning. The report records `corpus_tax_year` and flags any question whose ground truth names a different one, so the two drifting apart is visible rather than looking like poor retrieval.
- Answer relevancy at 0.633 is the weakest of the three metrics. Ragas scores it by generating questions from the answer and comparing them to the original, so verbose answers that cover more ground than was asked score lower.
- Column detection is a heuristic: it looks for a density trough in word coverage across the page. It falls back to single-column when it finds none, which is right for covers and full-width tables, but an unusual layout could still be split in the wrong place.
- Publications are ingested as static snapshots; re-ingest when IRS revises a publication (annual cycle for most).
- LangSmith tracing is opt-in, and it is the only path here that reaches a real advisory class. `LANGCHAIN_TRACING_V2` defaults to `false` in `docker-compose.yml`, so the deployed demo does not trace. With it `true`, the `@traceable` decorators in `embed.py`, `rerank.py` and `agent.py` send runs to LangSmith, and older SDK versions let streaming token events bypass output redaction. `pyproject.toml` pins `langsmith>=0.8.18` for that reason, so the path is patched rather than open. It is recorded here because it was the one advisory of thirteen that this project could actually reach: the rest name APIs it never calls (`load_prompt`, `ChatOpenAI`, `pull_prompt`, `TracingMiddleware`, `langgraph_sdk`, and an `HTMLHeaderTextSplitter` in a project that chunks its own PDFs), which is why a version-based scan overstated the exposure.

## License

MIT
