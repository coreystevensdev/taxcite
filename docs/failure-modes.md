# TaxCite failure modes

How each dependency fails, what you can actually see when it does, and what to do.
Identifiers here are checked against the code; where a detection channel does not
exist, this says so rather than inventing one.

The graph has five nodes: `retrieve`, `human_review`, `generate_answer`,
`no_documents`, `rejected` (`agent.py:200-204`). There is no separate embedding or
chunk-search node; `retrieve` does both.

## What you can see, before any of the specific modes

`GET /health` returns `{"status": "ok"}` unconditionally and touches nothing. It
tells you the process is up and nothing else, so it stays 200 through every
failure below.

`GET /status` is the useful one, and it carries three independent signals:

- `ingest` is `idle | running | ready | error`. The `error` field reflects the
  last ingest only, never the database.
- `chunks_by_pub` comes from a real query when `DATABASE_URL` is set. An empty map
  with a 200 means the corpus was never ingested; `count_chunks` failing is caught
  on purpose so this endpoint can report that rather than be the thing that fails.
- `answering` is `live` or `not_configured`, from the presence of the required API
  keys. Names only, never values. Check this before blaming the model for a 500.

One sharp edge worth knowing: `db.get_connection()` at `server.py:193` sits
*outside* that try block, so if the database is unreachable `/status` raises and
returns 500 itself. A 500 from `/status` is therefore a database signal, and an
empty `chunks_by_pub` with a 200 is not.

There are two tracers and they answer different questions. Reaching for the wrong
one is how you conclude a node failed when you cannot see nodes at all.

**LangSmith gives per-node spans.** `langsmith` is a real dependency and LangGraph
traces itself with no code in this repo, driven entirely by
`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY` and `LANGCHAIN_PROJECT`. This is the
only place you can see `retrieve` succeed and `generate_answer` fail. README
"LangSmith Tracing" has the setup and the shape of the trace tree.

**OpenTelemetry gives one span per HTTP request and nothing finer.**
`server.py:76-97` wires an OTLP exporter only when `OTEL_EXPORTER_OTLP_ENDPOINT`
is set, and the only instrumentation is `FastAPIInstrumentor.instrument_app(app)`.
No per-node spans come out of it, so a Jaeger trace cannot tell you which node
failed. `docker-compose.yml:31` runs the `jaeger` service it points at, which
means those spans exist in the compose stack and not in the deployed container.

**Both default to off.** `LANGCHAIN_TRACING_V2` defaults to `false` in compose and
is set to `false` explicitly in CI, and the OTLP endpoint defaults to a hostname
that only resolves inside compose. Turn one on before you need it, not after.

So unless tracing was enabled ahead of time, the log line is the detection
channel. Server-side handlers log with the `thread_id` before converting to an
HTTP status.

## 1. Claude is slow, rate limited, or over budget

`generate_answer` calls `client.messages.create` with **no timeout set**
(`agent.py:123`). A slow API call blocks the request for as long as the SDK's own
default allows. This is worth knowing before you go looking for a 90 second
ceiling that does not exist.

Three distinct outcomes, and they are not interchangeable:

- The cost cap trips first. `CostBudgetExceeded` is caught and logged as
  `Cost cap tripped for thread <id>`, and the caller gets **503** with
  "Service temporarily unavailable due to usage limits." Budget figures stay in
  the log and out of the response on purpose.
- A rate limit or transport error reaches the generic handler and is logged as
  `Graph execution failed for thread <id>` with a traceback.
- Nothing retries. There is no retry, no backoff and no circuit breaker anywhere
  in `src/`.

**Recovery:** retry `POST /ask` with a new `thread_id`. Resuming the old thread
re-enters at `human_review`, not at the failed generation. For a tripped cost cap,
the window has to pass or the budget has to be raised; retrying immediately gets
the same 503.

## 2. Postgres is unreachable

`retrieve` embeds the question and searches the `chunks` table, whose index is
**HNSW** (`db.py:46`, `ON chunks USING hnsw (embedding vector_cosine_ops)`). An
earlier version used IVFFlat and that was a defect, not a design: the index is
created at migration time before any rows exist, and IVFFlat derives its centroids
from the data present at build time, so it was trained on an empty table. HNSW
builds incrementally and does not care. Do not reintroduce IVFFlat here.

**Detection:** `psycopg2.OperationalError` is caught specifically and logged as
`Database unavailable for thread <id>`; the caller gets **503** with "Database
unavailable. Retry shortly." `/health` is still 200. `/status` returns 500, for the
reason above; its `error` field stays silent because that field is about ingest.

**Recovery:** bring the database back. The index is part of the table and comes
back with it, so there is nothing to rebuild by hand. If the data itself is
suspect, re-ingest (section 3); `upsert_chunk` is an upsert, so re-ingesting over
existing rows is safe.

## 3. Voyage rate limits, during ingest or a query

Both paths call the same API. `embed_texts` during ingest, `embed_query` at
request time, both `voyage-3.5-lite` (`embed.py:11`). The Ragas judge embeds with
`voyage-3`, which is a different model and only used by the eval.

**Recovery, ingest:** re-run it. `python -m taxcite ingest` takes optional
publication ids, so `python -m taxcite ingest p501` redoes one. It is safe to
repeat: `db.py:113` is `ON CONFLICT (pub_id, ordinal) DO UPDATE SET`, so a second
run refreshes rows rather than skipping them, and `prune_chunks` drops any
trailing chunks a shorter re-parse leaves behind. Each publication commits as a
unit and rolls back on a `psycopg2.Error`, so a mid-run failure leaves whole
publications, not half of one.

**Recovery, query:** retry after the per-minute window. Do not switch embedding
models to get around it. Stored vectors are 1024-dimensional and a different model
either mismatches the column or silently changes what "similar" means.

## 4. A human review that can never be resumed

`server.py:137` is `_checkpointer = MemorySaver()` at module scope. Interrupt and
resume therefore work **within one process and only while it lives**.

Two ways this bites:

- The process restarts while a thread sits at `human_review`. The checkpoint is
  gone, and resuming that `thread_id` cannot find the state. A deploy does this to
  every pending review.
- More than one replica. A resume routed to a different process than the one
  holding the checkpoint fails the same way, non-deterministically.

**Detection:** a resume for a `thread_id` that the checkpointer does not have.
This looks like a client error and is not one.

**Recovery:** start over with a new `POST /ask`. The real fix is a
`PostgresSaver` on the same database, which `agent.py:194` and the comment at
`server.py:136` both name; it is not wired up, so do not describe the deployment
as durable.

## 5. Retrieval finds nothing

Not a failure. `_route_after_retrieve` sends an empty result to the
`no_documents` node, which answers that no relevant excerpts were found. The
pipeline refusing to answer is the designed behavior and the reason faithfulness
scores as high as it does.

It becomes a failure when it is *wrong*, and the known cause is a corpus and
dataset that disagree about the tax year. `manifest.TAX_YEAR` pins the corpus and
each eval ground truth states its own year; move one without the other and the
agent correctly refuses to state a year it does not have, which the eval scores as
zero. `corpus_tax_year` and `_pinned_year` in `eval_harness.py` exist to separate
that from a real failure.

## 6. Quality regresses without a test failing

A retrieval or prompt change can degrade answers while every test passes.

**Detection:** `python -m taxcite eval`, then compare against the scores in the
README. Read the per-question scores, not only the means: the report keeps every
question's own scores, which is what makes a retrieval change a paired comparison
instead of two means that might differ by noise.

**Before comparing anything, check provenance.** A run records `provenance` and
`score_coverage`; `eval/report.json` as committed is from 2026-09-11 and predates
both fields, so it carries no model version or code revision and cannot serve as a
baseline for a later run. Generate a fresh baseline before attributing a
difference to your change. That ambiguity has already cost one full re-run.

**Recovery:** roll the change back and re-run. If a change is worth keeping at a
lower score, update the README's numbers and say why in the same commit.
