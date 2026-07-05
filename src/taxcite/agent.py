"""LangGraph agent: retrieve IRS publication chunks and generate cited answers."""
from __future__ import annotations

import os
from typing import TypedDict

import anthropic
from langsmith import traceable
from langgraph.graph import END, StateGraph

from taxcite import db, embed
from taxcite.chunk import Chunk

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_CHUNKS = 8

_SUBMIT_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit the final answer and deduplicated source citations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Full answer with inline citations like [Pub 936, pp.12-13].",
            },
            "citations": {
                "type": "array",
                "description": "Deduplicated list of IRS pub citations used in the answer.",
                "items": {
                    "type": "object",
                    "properties": {
                        "pub_id": {"type": "string"},
                        "first_page": {"type": "integer"},
                        "last_page": {"type": "integer"},
                    },
                    "required": ["pub_id", "first_page", "last_page"],
                },
            },
        },
        "required": ["answer", "citations"],
    },
}

_SYSTEM_PROMPT = (
    "You are a tax research assistant. Answer questions using ONLY the provided "
    "IRS publication excerpts. Cite each fact with [pub_id, pages] inline. "
    "If the excerpts do not contain enough information to answer fully, say so clearly."
)


class AgentState(TypedDict):
    question: str
    chunks: list[Chunk]
    answer: str
    citations: list[dict]  # {pub_id, first_page, last_page}


def retrieve(state: AgentState) -> dict:
    query_vec = embed.embed_query(state["question"])
    conn = db.get_connection()
    try:
        chunks = db.search_chunks(conn, query_vec, top_k=MAX_CHUNKS)
    finally:
        conn.close()
    return {"chunks": chunks}


@traceable(name="generate_answer", run_type="llm")
def generate_answer(state: AgentState) -> dict:
    context_blocks = []
    for i, chunk in enumerate(state["chunks"], 1):
        if chunk.first_page == chunk.last_page:
            pages = f"p.{chunk.first_page}"
        else:
            pages = f"pp.{chunk.first_page}-{chunk.last_page}"
        context_blocks.append(f"[{i}] {chunk.pub_id}, {pages}:\n{chunk.text}")

    user_content = (
        f"Question: {state['question']}\n\n"
        f"IRS Publication Excerpts:\n\n"
        + "\n\n".join(context_blocks)
        + "\n\nAnswer the question with inline citations, then call submit_answer."
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        tools=[_SUBMIT_ANSWER_TOOL],
        tool_choice={"type": "tool", "name": "submit_answer"},
    )

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_answer":
            return {
                "answer": block.input["answer"],
                "citations": block.input.get("citations", []),
            }

    # Should not reach here with tool_choice forced, but guard defensively.
    text = next((b.text for b in response.content if hasattr(b, "text")), "")
    return {"answer": text, "citations": []}


def no_documents(state: AgentState) -> dict:  # noqa: ARG001
    return {
        "answer": (
            "No relevant IRS publication excerpts were found for this question. "
            "Ensure the relevant publications have been ingested with `taxcite ingest`."
        ),
        "citations": [],
    }


def _route_after_retrieve(state: AgentState) -> str:
    return "generate_answer" if state["chunks"] else "no_documents"


def build_graph():
    """Compile and return the LangGraph agent."""
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("no_documents", no_documents)

    graph.set_entry_point("retrieve")

    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"generate_answer": "generate_answer", "no_documents": "no_documents"},
    )

    graph.add_edge("generate_answer", END)
    graph.add_edge("no_documents", END)

    return graph.compile()
