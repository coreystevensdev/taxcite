"""Ragas eval harness for TaxCite.

Runs the agent on a JSONL dataset, scores with Ragas faithfulness /
answer_relevancy / context_precision, and writes a JSON report.

Usage:
    python -m taxcite eval [--dataset eval/dataset.jsonl] [--out eval/report.json]

Requires the eval extras (ragas, datasets) and a live DATABASE_URL with
ingested publications. Ragas scoring uses the LLM configured via the
OPENAI_API_KEY environment variable (default) or a custom judge via
RAGAS_LLM env var (see ragas docs).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_DATASET = Path(__file__).parent.parent.parent / "eval" / "dataset.jsonl"
DEFAULT_REPORT = Path(__file__).parent.parent.parent / "eval" / "report.json"


def _load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _run_agent_on_question(graph, question: str) -> tuple[str, list[str]]:
    """Invoke the compiled LangGraph and return (answer, context_texts)."""
    from taxcite.agent import AgentState

    initial: AgentState = {
        "question": question,
        "chunks": [],
        "answer": "",
        "citations": [],
    }
    final = graph.invoke(initial)
    contexts = [c.text for c in final.get("chunks", [])]
    return final["answer"], contexts


def _score_with_ragas(records: list[dict]) -> dict:
    """Score records with Ragas; returns metric dict + per-question scores."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        raise SystemExit(
            "Install eval extras first: pip install taxcite[eval]"
        ) from exc

    dataset = Dataset.from_list(records)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    return result


def run_eval(dataset_path: Path = DEFAULT_DATASET, report_path: Path = DEFAULT_REPORT) -> dict:
    from taxcite.agent import build_graph

    items = _load_dataset(dataset_path)
    graph = build_graph()

    records = []
    for item in items:
        answer, contexts = _run_agent_on_question(graph, item["question"])
        records.append(
            {
                "question": item["question"],
                "answer": answer,
                "contexts": contexts if contexts else [""],
                "ground_truth": item.get("ground_truth", ""),
            }
        )
        print(f"  answered: {item['question'][:60]}")

    print("Scoring with Ragas...")
    scores = _score_with_ragas(records)

    report = {
        "metrics": {k: round(float(v), 4) for k, v in scores.items() if isinstance(v, (int, float))},
        "n_questions": len(items),
        "per_question": records,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {report_path}")
    print("Scores:", report["metrics"])
    return report
