"""FastAPI server exposing the TaxCite agent over HTTP."""
from __future__ import annotations

import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import anthropic
import psycopg2
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
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

# Set when Anthropic refuses on balance, cleared by the next generation that
# works. /status needs it because key presence cannot tell a configured
# deployment from a funded one, and on 2026-11-22 that gap let the endpoint
# report "live" through an outage where every answer 500'd.
#
# A dict for the same reason _ingest is one: the routes mutate it, and a plain
# module-level bool would need `global` in each of them to not quietly become a
# local that nothing reads.
_billing: dict[str, bool] = {"blocked": False}

# Set when a cost cap refuses a call, cleared by the next one that completes.
# Separate from _billing because the two are different outages with the same
# 503: a funded account whose cap has tripped is not an unfunded one, and the
# remedies are opposite. /status reported "live" through eight cap trips on
# 2026-11-25 because `answering` had no state for this, which is the exact
# failure the comment above it warns about one outage earlier.
_capped: dict[str, bool] = {"tripped": False}

# Anthropic gives an exhausted balance no distinct error type or code: it is a
# BadRequestError whose message names it. Matching the text is fragile, so a miss
# falls through to the behaviour this had before rather than to a wrong answer.
_CREDIT_EXHAUSTED = "credit balance is too low"


def _is_credit_exhausted(exc: anthropic.BadRequestError) -> bool:
    return _CREDIT_EXHAUSTED in str(exc)


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
        except psycopg2.Error:
            # Before the first ingest the chunks table does not exist. This is
            # the endpoint you check to find that out, so it reports an empty
            # corpus rather than being the thing that fails.
            logger.info("chunk counts unavailable; corpus not ingested yet")
        finally:
            db.release_connection(conn)

    # The page needs to know this before it renders a control that may not work.
    # Names only, never values: this endpoint is public.
    if not all(os.environ.get(k) for k in _REQUIRED_KEYS):
        answering = "not_configured"
    elif _billing["blocked"]:
        # Observed, not probed: checking a balance costs a call, so this reports
        # what the last generation actually got back.
        answering = "no_credits"
    elif _capped["tripped"]:
        # After no_credits on purpose. Both can be set at once, and an exhausted
        # balance outranks a tripped cap: it blocks every call rather than one
        # workload, and topping up is a different remedy from widening a cap.
        # Reporting the cap while the account is empty would send a reader to
        # fix the wrong thing.
        answering = "capped"
    else:
        answering = "live"

    body: dict = {
        "ingest": _ingest["state"],
        "chunks_by_pub": counts,
        "answering": answering,
    }
    if "stage" in _ingest:
        body["stage"] = _ingest["stage"]
    if "error" in _ingest:
        # The raw cause (exception type + message, possibly a DSN or path) is in
        # the server log; the public endpoint gets a coarse flag only.
        body["error"] = "ingest failed; see server logs"
    return body


# Both vendors are read deep inside graph nodes, where a missing key surfaces as
# a KeyError and reaches the caller as a 500 with a stack trace. Checked up front
# instead, so a deployment without them says so rather than looking broken.
_REQUIRED_KEYS = ("VOYAGE_API_KEY", "ANTHROPIC_API_KEY")


def _require_vendor_keys() -> None:
    missing = [k for k in _REQUIRED_KEYS if not os.environ.get(k)]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=(
                "Question answering is not configured on this deployment: "
                + ", ".join(missing)
                + " is unset. The evaluation report at / needs neither and is unaffected."
            ),
        )


@app.post("/ask")
@limiter.limit("10/minute")
def ask(request: Request, req: AskRequest) -> AskCompleteResponse | AskAwaitingResponse:
    _require_vendor_keys()
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
        _capped["tripped"] = True
        logger.warning("Cost cap tripped for thread %s: %s", thread_id, exc)
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable due to usage limits. Retry later.",
        ) from exc
    except anthropic.BadRequestError as exc:
        # Nothing caught this before: BadRequestError descends from Exception,
        # not ValueError or RuntimeError, so it reached FastAPI's default
        # handler and a visitor saw a bare "Internal Server Error".
        if not _is_credit_exhausted(exc):
            raise
        _billing["blocked"] = True
        logger.error("Anthropic refused on balance for thread %s", thread_id)
        raise HTTPException(
            status_code=503,
            detail="Answer generation is unavailable: this deployment is out of model credits.",
        ) from exc
    except psycopg2.OperationalError as exc:
        # DB is unreachable or down: a dependency failure, not a server bug.
        logger.exception("Database unavailable for thread %s", thread_id)
        raise HTTPException(status_code=503, detail="Database unavailable. Retry shortly.") from exc
    except (ValueError, RuntimeError, psycopg2.Error) as exc:
        # Covers: malformed state, LangGraph compile errors, in-flight DB query errors.
        logger.exception("Graph execution failed for thread %s", thread_id)
        raise HTTPException(status_code=500, detail="Graph execution failed. Check server logs.") from exc

    _billing["blocked"] = False
    # This path embedded the question, so the embed caps are demonstrably
    # passing. Unlike the billing flag it is not claiming anything about
    # Anthropic, which /ask never reaches.
    _capped["tripped"] = False
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
        _capped["tripped"] = True
        logger.warning("Cost cap tripped for thread %s: %s", req.thread_id, exc)
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable due to usage limits. Retry later.",
        ) from exc
    except anthropic.BadRequestError as exc:
        # Nothing caught this before: BadRequestError descends from Exception,
        # not ValueError or RuntimeError, so it reached FastAPI's default
        # handler and a visitor saw a bare "Internal Server Error".
        if not _is_credit_exhausted(exc):
            raise
        _billing["blocked"] = True
        logger.error("Anthropic refused on balance for thread %s", req.thread_id)
        raise HTTPException(
            status_code=503,
            detail="Answer generation is unavailable: this deployment is out of model credits.",
        ) from exc
    except psycopg2.OperationalError as exc:
        # DB is unreachable or down: a dependency failure, not a server bug.
        logger.exception("Database unavailable for thread %s", req.thread_id)
        raise HTTPException(status_code=503, detail="Database unavailable. Retry shortly.") from exc
    except (ValueError, RuntimeError, psycopg2.Error) as exc:
        # Covers: bad resume state, LangGraph internal errors, in-flight DB query errors.
        logger.exception("Graph resume failed for thread %s", req.thread_id)
        raise HTTPException(status_code=500, detail="Graph resume failed. Check server logs.") from exc

    _billing["blocked"] = False
    _capped["tripped"] = False
    values = _graph.get_state(config).values
    return AskCompleteResponse(
        answer=values.get("answer", ""),
        citations=[Citation(**c) for c in values.get("citations", [])],
    )


_WEB = Path(__file__).resolve().parent / "web"
_PAGE = _WEB / "index.html"


def _find_report() -> Path:
    """Locate eval/report.json whether running from the repo or the image.

    Installed, __file__ sits in site-packages and walking up from it lands
    nowhere near the report, which the Dockerfile copies next to the working
    directory. The repo layout made the local test pass and the container 404.
    """
    override = os.environ.get("TAXCITE_EVAL_REPORT")
    candidates = [Path(override)] if override else []
    candidates += [
        Path.cwd() / "eval" / "report.json",
        Path(__file__).resolve().parent.parent.parent / "eval" / "report.json",
    ]
    return next((c for c in candidates if c.is_file()), candidates[-1])

# The report is ~700KB of mostly prose, so it compresses to a fraction of that.
app.add_middleware(GZipMiddleware, minimum_size=1024)


# The page used to carry both of these inline, which is why it could not have a
# Content-Security-Policy worth setting: any policy permitting its own <script>
# also permits an injected one. Nine innerHTML assignments render report prose,
# so that was the surface worth closing.
#
# Served as files rather than nonced inline: a nonce has to change per response,
# the page is a FileResponse off disk, and templating it just to vary one
# attribute buys nothing that 'self' does not already give. Files also cache.
@app.get("/app.css", include_in_schema=False)
async def app_css():
    return FileResponse(_WEB / "app.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
async def app_js():
    return FileResponse(_WEB / "app.js", media_type="application/javascript")


# Built once. style-src keeps 'unsafe-inline' because five style= attributes and
# one element.style assignment remain in the markup, and style injection is a far
# smaller problem than script injection; script-src carries no such escape.
# Google Fonts needs the stylesheet host on style-src and the font host on
# font-src, which is the one pair that cannot be 'self'.
_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
    "object-src 'none'"
)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Content-Security-Policy", _CSP)
    return response


@app.get("/", include_in_schema=False)
async def eval_page():
    """The browsable eval report.

    Nothing here is generated: the page reads the committed report and renders
    it. An ask box would have been the obvious demo and would have shown less,
    since it hides retrieval and hides whether the answer was any good.
    """
    return FileResponse(_PAGE, media_type="text/html")


@app.get("/eval/report.json", include_in_schema=False)
async def eval_report():
    """Served from the file rather than rebuilt, so a reader can diff what the
    page shows against what the repository contains."""
    report = _find_report()
    if not report.is_file():
        raise HTTPException(status_code=404, detail="eval report not found")
    return FileResponse(report, media_type="application/json")
