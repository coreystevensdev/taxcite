"""FastAPI server exposing the TaxCite agent over HTTP."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from taxcite.agent import AgentState, build_graph

app = FastAPI(title="TaxCite", version="0.1.0")

# Compiled once at import time; shared across requests (stateless graph).
_graph = build_graph()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Citation(BaseModel):
    pub_id: str
    first_page: int
    last_page: int


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    initial: AgentState = {
        "question": req.question,
        "chunks": [],
        "answer": "",
        "citations": [],
    }
    try:
        final = _graph.invoke(initial)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    citations = [
        Citation(
            pub_id=c["pub_id"],
            first_page=c["first_page"],
            last_page=c["last_page"],
        )
        for c in final.get("citations", [])
    ]
    return AskResponse(answer=final["answer"], citations=citations)
