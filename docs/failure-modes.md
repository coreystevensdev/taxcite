# TaxCite Failure Modes

Forward-looking documentation of how each system layer can fail, how to detect it, and how to recover.

## 1. Claude API timeout or rate limit

**Scenario:** The `generate_answer` node calls `anthropic.messages.create` with a 90-second timeout. If the API is slow or the account hits rate limits, the call raises `httpx.TimeoutException` or `anthropic.RateLimitError`.

**Detection:** LangSmith trace shows the `generate_answer` step incomplete. Jaeger span shows a long gap or missing child span. HTTP 500 returned to caller.

**Recovery:** The LangGraph agent does not retry automatically (retrying a partial RAG answer could surface inconsistent results). The caller should retry `POST /ask` with a new `thread_id`. For sustained rate limits, back off and alert on `anthropic.RateLimitError` in the LangSmith trace.

## 2. pgvector index corruption or OOM

**Scenario:** The `retrieve_chunks` node runs an approximate nearest-neighbor query against the `documents` table. If the IVFFlat index is corrupted or Postgres OOMs mid-query, the connection raises `psycopg2.OperationalError`.

**Detection:** `GET /health` still returns 200 (it does not query pgvector). The failure surfaces only on `POST /ask`. Monitor for `psycopg2.OperationalError` in logs. Jaeger shows the retrieve span failing with a DB error.

**Recovery:** Restart the Postgres service. The IVFFlat index rebuilds on restart (it is not persisted separately from the table). If index corruption is persistent, drop and re-ingest: `python -m taxcite.ingest`.

## 3. Voyage AI rate limit

**Scenario:** During ingest, `voyageai.Client.embed` calls are subject to Voyage's token-per-minute limit. During query time, the `embed_question` node calls the same API. If the limit is hit, `voyageai.error.RateLimitError` is raised.

**Detection:** Ingest fails partway through with a rate-limit error logged to stderr. At query time, `POST /ask` returns 500 with the Voyage error in the trace.

**Recovery:** For ingest, re-run `taxcite.ingest` starting from the failed publication (idempotent: chunks are inserted with `ON CONFLICT DO NOTHING`). For query time, retry after the per-minute window resets (60 seconds). Do not switch to a different embedding model mid-deployment: vector dimensions would mismatch the stored chunks.

## 4. Render cold start (spindown)

**Scenario:** Render's free tier spins down instances after 15 minutes of inactivity. The next request triggers a cold start of 30-60 seconds.

**Detection:** The first `POST /ask` after a spindown period returns a 502 or takes 30+ seconds. Subsequent requests are normal.

**Recovery:** No action needed; this is expected behavior on the free tier. For demos: hit `GET /health` before the demo to wake the instance. For production use: upgrade to a paid Render plan or move to a persistent host.

## 5. Ragas regression

**Scenario:** A change to the retrieval pipeline (chunk size, overlap, embedding model, HITL threshold) silently degrades answer quality without causing test failures.

**Detection:** Run `python -m taxcite.eval` after any retrieval or generation change. Check faithfulness, answer_relevancy, and context_precision against the baseline scores in the README. A drop of more than 0.05 on any metric warrants investigation.

**Recovery:** Roll back the retrieval or prompt change. Re-run eval to confirm scores recover. Do not commit a change that degrades Ragas scores without updating the README with the new baseline and a documented reason.

## 6. LangSmith auth failure

**Scenario:** `LANGCHAIN_API_KEY` is missing or rotated. `LANGCHAIN_TRACING_V2=true` is set. LangGraph tries to upload traces and fails with an auth error.

**Detection:** Warning logged at startup: "Failed to connect to LangSmith". Traces do not appear in the LangSmith dashboard. The agent still runs; tracing is non-blocking.

**Recovery:** Set `LANGCHAIN_TRACING_V2=false` to disable tracing until the key is updated. Rotate the key in the LangSmith dashboard and update the environment variable.
