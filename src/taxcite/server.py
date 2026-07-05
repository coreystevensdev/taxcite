"""FastAPI server exposing the TaxCite agent over HTTP."""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from taxcite.agent import AgentState, build_graph


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
    # Idempotent migration: safe to run on every cold start.
    if os.getenv("DATABASE_URL"):
        from taxcite import db
        conn = db.get_connection()
        try:
            db.run_migration(conn)
        finally:
            conn.close()
    yield


app = FastAPI(title="TaxCite", version="0.1.0", lifespan=lifespan)
_configure_telemetry(app)

# MemorySaver enables HITL interrupt/resume across requests (per-process).
# In a multi-replica deployment replace with PostgresSaver backed by the same DB.
_checkpointer = MemorySaver()
_graph = build_graph(checkpointer=_checkpointer)


# --- Request / response models ---

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
    thread_id: str
    approved: bool


# --- Routes ---

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest) -> AskCompleteResponse | AskAwaitingResponse:
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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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
def ask_resume(req: ResumeRequest) -> AskCompleteResponse:
    """Resume a graph run paused at the human_review interrupt."""
    config = {"configurable": {"thread_id": req.thread_id}}

    snapshot = _graph.get_state(config)
    if not snapshot or not snapshot.next:
        raise HTTPException(status_code=404, detail="No interrupted run found for this thread_id.")

    try:
        _graph.invoke(Command(resume=req.approved), config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    values = _graph.get_state(config).values
    return AskCompleteResponse(
        answer=values.get("answer", ""),
        citations=[Citation(**c) for c in values.get("citations", [])],
    )
