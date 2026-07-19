"""FastAPI server exposing the TaxCite agent over HTTP."""
from __future__ import annotations

import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import psycopg2

from fastapi import FastAPI, HTTPException, Request
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from taxcite.agent import AgentState, build_graph
from taxcite.cost import CostBudgetExceeded

logger = logging.getLogger(__name__)

# Tracks background ingest state so /status can surface it.
_ingest: dict[str, object] = {"state": "idle"}


def _run_ingest_thread() -> None:
    """Populate the vector store from the IRS corpus. Runs once in a daemon thread."""
    from taxcite import db
    from taxcite.chunk import chunk_pages
    from taxcite.embed import embed_texts
    from taxcite.fetch import fetch_publication
    from taxcite.manifest import CORPUS
    from taxcite.parse import parse_pdf

    _ingest["state"] = "running"
    _ingest.pop("error", None)
    logger.info("auto-ingest: starting")
    conn = db.get_connection()
    try:
        for pub in CORPUS:
            _ingest["stage"] = pub.pub_id
            path = fetch_publication(pub)
            pages = parse_pdf(path)
            chunks = chunk_pages(pub.pub_id, pages)
            embeddings = embed_texts([c.text for c in chunks])
            try:
                for chunk, embedding in zip(chunks, embeddings):
                    db.upsert_chunk(conn, chunk, embedding, commit=False)
                db.prune_chunks(conn, pub.pub_id, len(chunks), commit=False)
                conn.commit()
            except psycopg2.Error:
                conn.rollback()
                raise
            logger.info("auto-ingest: %s done (%d chunks)", pub.pub_id, len(chunks))
        _ingest["state"] = "ready"
        _ingest.pop("stage", None)
        logger.info("auto-ingest: complete")
    except (RuntimeError, OSError, psycopg2.Error) as exc:
        # Covers FetchError/EmbeddingError/CostBudgetExceeded (RuntimeError
        # subclasses), pdfplumber file I/O (OSError), and DB errors.
        _ingest["state"] = "error"
        # Surface the cause at the thread boundary so /status can report which stage failed.
        _ingest["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("auto-ingest: failed at stage %s", _ingest.get("stage"))
    finally:
        db.release_connection(conn)


def _configure_telemetry(app: FastAPI) -> None:
    """Wire OpenTelemetry OTLP tracing when OTEL_EXPORTER_OTLP_ENDPOINT is set.

    Traces are sent to a local Jaeger all-in-one instance by default.
    Set OTEL_EXPORTER_OTLP_ENDPOINT to override (e.g. a hosted collector).
    When the env var is absent, OTEL is a no-op and startup is unaffected.
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "taxcite")})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("DATABASE_URL"):
        from taxcite import db

        # Schema migration is idempotent; safe on every cold start.
        conn = db.get_connection()
        try:
            db.run_migration(conn)
            chunk_counts = db.count_chunks(conn)
        finally:
            db.release_connection(conn)

        # On first boot (empty DB), populate the corpus in the background.
        # The server returns /health 200 immediately so Render's healthcheck
        # doesn't time out. Queries return empty results until ingest completes.
        if not chunk_counts and os.getenv("VOYAGE_API_KEY"):
            t = threading.Thread(target=_run_ingest_thread, daemon=True)
            t.start()
        else:
            _ingest["state"] = "ready" if chunk_counts else "idle"
    yield
    if os.getenv("DATABASE_URL"):
        from taxcite import db

        db.close_pool()


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="TaxCite", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
_configure_telemetry(app)

# MemorySaver enables HITL interrupt/resume across requests (per-process).
# In a multi-replica deployment replace with PostgresSaver backed by the same DB.
_checkpointer = MemorySaver()
_graph = build_graph(checkpointer=_checkpointer)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Citation(BaseModel):
    pub_id: str
    first_page: int
    last_page: int


class ChunkPreview(BaseModel):
    pub_id: str
    pages: str
    excerpt: str


class AskCompleteResponse(BaseModel):
    status: Literal["complete"] = "complete"
    answer: str
    citations: list[Citation]


class AskAwaitingResponse(BaseModel):
    status: Literal["awaiting_review"] = "awaiting_review"
    thread_id: str
    chunks_preview: list[ChunkPreview]


class ResumeRequest(BaseModel):
    thread_id: str = Field(..., min_length=36, max_length=36)
    approved: bool

    @field_validator("thread_id")
    @classmethod
    def _must_be_uuid(cls, v: str) -> str:
        # thread_ids are server-minted UUIDs; reject anything else before it
        # reaches the checkpointer instead of probing arbitrary state keys.
        uuid.UUID(v)
        return v


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict:
    """Ingest state: idle | running | ready | error. Ready means /ask is functional."""
    counts: dict = {}
    if os.getenv("DATABASE_URL"):
        from taxcite import db
        conn = db.get_connection()
        try:
            counts = db.count_chunks(conn)
        finally:
            db.release_connection(conn)

    body: dict = {"ingest": _ingest["state"], "chunks_by_pub": counts}
    if "stage" in _ingest:
        body["stage"] = _ingest["stage"]
    if "error" in _ingest:
        # The raw cause (exception type + message, possibly a DSN or path) is in
        # the server log; the public endpoint gets a coarse flag only.
        body["error"] = "ingest failed; see server logs"
    return body


@app.post("/ask")
@limiter.limit("10/minute")
def ask(request: Request, req: AskRequest) -> AskCompleteResponse | AskAwaitingResponse:
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial: AgentState = {
        "question": req.question,
        "chunks": [],
        "answer": "",
        "citations": [],
    }
    try:
        _graph.invoke(initial, config=config)
    except CostBudgetExceeded as exc:
        # Log the spend detail; don't hand budget figures to an anonymous caller.
        logger.warning("Cost cap tripped for thread %s: %s", thread_id, exc)
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable due to usage limits. Retry later.",
        ) from exc
    except psycopg2.OperationalError as exc:
        # DB is unreachable or down: a dependency failure, not a server bug.
        logger.exception("Database unavailable for thread %s", thread_id)
        raise HTTPException(status_code=503, detail="Database unavailable. Retry shortly.") from exc
    except (ValueError, RuntimeError, psycopg2.Error) as exc:
        # Covers: malformed state, LangGraph compile errors, in-flight DB query errors.
        logger.exception("Graph execution failed for thread %s", thread_id)
        raise HTTPException(status_code=500, detail="Graph execution failed. Check server logs.") from exc

    snapshot = _graph.get_state(config)

    if snapshot.next:
        # Graph is paused at the human_review interrupt.
        chunks = snapshot.values.get("chunks", [])
        return AskAwaitingResponse(
            thread_id=thread_id,
            chunks_preview=[
                ChunkPreview(
                    pub_id=c.pub_id,
                    pages=f"pp.{c.first_page}-{c.last_page}" if c.first_page != c.last_page else f"p.{c.first_page}",
                    excerpt=c.text[:300],
                )
                for c in chunks
            ],
        )

    values = snapshot.values
    return AskCompleteResponse(
        answer=values.get("answer", ""),
        citations=[Citation(**c) for c in values.get("citations", [])],
    )


@app.post("/ask/resume")
@limiter.limit("20/minute")
def ask_resume(request: Request, req: ResumeRequest) -> AskCompleteResponse:
    """Resume a graph run paused at the human_review interrupt."""
    config = {"configurable": {"thread_id": req.thread_id}}

    snapshot = _graph.get_state(config)
    if not snapshot or not snapshot.next:
        raise HTTPException(status_code=404, detail="No interrupted run found for this thread_id.")

    try:
        _graph.invoke(Command(resume=req.approved), config=config)
    except CostBudgetExceeded as exc:
        logger.warning("Cost cap tripped for thread %s: %s", req.thread_id, exc)
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable due to usage limits. Retry later.",
        ) from exc
    except psycopg2.OperationalError as exc:
        # DB is unreachable or down: a dependency failure, not a server bug.
        logger.exception("Database unavailable for thread %s", req.thread_id)
        raise HTTPException(status_code=503, detail="Database unavailable. Retry shortly.") from exc
    except (ValueError, RuntimeError, psycopg2.Error) as exc:
        # Covers: bad resume state, LangGraph internal errors, in-flight DB query errors.
        logger.exception("Graph resume failed for thread %s", req.thread_id)
        raise HTTPException(status_code=500, detail="Graph resume failed. Check server logs.") from exc

    values = _graph.get_state(config).values
    return AskCompleteResponse(
        answer=values.get("answer", ""),
        citations=[Citation(**c) for c in values.get("citations", [])],
    )
