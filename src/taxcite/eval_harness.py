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
import re
from pathlib import Path

DEFAULT_DATASET = Path(__file__).parent.parent.parent / "eval" / "dataset.jsonl"
DEFAULT_REPORT = Path(__file__).parent.parent.parent / "eval" / "report.json"


def _load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _run_agent_on_question(graph, question: str, config: dict) -> tuple[str, list[str]]:
    """Invoke the compiled LangGraph and return (answer, context_texts).

    Auto-approves the HITL interrupt so the eval runs end-to-end without
    human input. Uses the same get_state / Command(resume=True) pattern as
    the server's /ask + /ask/resume flow.
    """
    from langgraph.types import Command

    from taxcite.agent import AgentState

    initial: AgentState = {
        "question": question,
        "chunks": [],
        "answer": "",
        "citations": [],
    }
    graph.invoke(initial, config=config)

    snapshot = graph.get_state(config)
    if snapshot.next:
        # Paused at human_review: auto-approve for eval.
        graph.invoke(Command(resume=True), config=config)
        snapshot = graph.get_state(config)

    values = snapshot.values
    contexts = [c.text for c in values.get("chunks", [])]
    return values.get("answer", ""), contexts


METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")


def _score_with_ragas(records: list[dict]):
    """Run Ragas and return its EvaluationResult."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        raise SystemExit(
            "Install eval extras first: pip install taxcite[eval]"
        ) from exc

    dataset = Dataset.from_list(records)
    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=_judge_llm(),
        embeddings=_judge_embeddings(),
    )


def _judge_llm():
    """Claude as the judge, rather than whatever Ragas reaches for by default.

    Left unset, `evaluate` builds an OpenAI client and fails on auth, which is a
    third provider this project neither configures nor needs. `.env.example` still
    lists OPENAI_API_KEY for that reason and the key on file was an Anthropic one,
    so the eval could not have run as shipped.
    """
    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            temperature=0,
            max_tokens=8192,
        )
    )


def _judge_embeddings():
    """answer_relevancy embeds the generated question, so Ragas needs embeddings
    of its own. Voyage again, so the judge measures distance the same way
    retrieval does."""
    from langchain_voyageai import VoyageAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(VoyageAIEmbeddings(model="voyage-3"))


# A ground truth that opens "For 2024, ..." is pinned to a tax year. The IRS
# serves every publication from an unversioned URL, so the same ingest command
# fetches a different revision each January and a dataset written against last
# year's figures starts scoring correct refusals as failures.
_YEAR_IN_REFERENCE = re.compile(r"\bFor (20\d{2})\b|\b(20\d{2}) tax year\b")


def _pinned_year(reference: str) -> str | None:
    match = _YEAR_IN_REFERENCE.search(reference[:80])
    if not match:
        return None
    return match.group(1) or match.group(2)


def corpus_tax_year(chunks_text: list[str]) -> str | None:
    """The year the ingested publications are actually about, by weight of mention."""
    counts: dict[str, int] = {}
    for text in chunks_text:
        for year in re.findall(r"\b20[2-9]\d\b", text):
            counts[year] = counts.get(year, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _aggregate_metrics(scores) -> dict[str, float]:
    """Mean each metric column of a Ragas EvaluationResult.

    ragas >= 0.2 returns an EvaluationResult, not a dict: per-sample scores
    live in .to_pandas() and the headline number is the column mean.
    """
    df = scores.to_pandas()
    return {
        name: round(float(df[name].mean()), 4)
        for name in METRIC_NAMES
        if name in df.columns
    }


def _aggregate_metrics_for(scores, rows: list[int]) -> dict[str, float]:
    """Means over a subset of questions, by row index.

    Reported alongside the headline so a reader can see what the corpus scores on
    the questions it could actually answer, without that replacing the real
    number. Both belong in the report: the lower one is what the system does
    against this dataset today, the higher one is what it does when the dataset
    and the ingested revision agree.
    """
    df = scores.to_pandas()
    if not rows:
        return {}
    subset = df.iloc[rows]
    return {
        name: round(float(subset[name].mean()), 4)
        for name in METRIC_NAMES
        if name in subset.columns
    }


def _per_sample_metrics(scores) -> list[dict[str, float]]:
    """Each question's own scores, not just the column means.

    Two runs of this harness differ by ordinary model variance, so comparing a
    retrieval change against a baseline on aggregate numbers alone cannot separate
    a real gain from noise. Keeping the per-question scores makes that a paired
    comparison over the same 50 questions, which is a far sharper instrument and
    costs nothing extra, the numbers are already computed.
    """
    df = scores.to_pandas()
    present = [n for n in METRIC_NAMES if n in df.columns]
    return [
        {name: (None if (v := row[name]) != v else round(float(v), 4)) for name in present}
        for _, row in df.iterrows()
    ]


def run_eval(dataset_path: Path = DEFAULT_DATASET, report_path: Path = DEFAULT_REPORT) -> dict:
    from langgraph.checkpoint.memory import MemorySaver

    from taxcite.agent import build_graph

    items = _load_dataset(dataset_path)
    graph = build_graph(checkpointer=MemorySaver())

    records = []
    for i, item in enumerate(items):
        config = {"configurable": {"thread_id": f"eval-{i}"}}
        answer, contexts = _run_agent_on_question(graph, item["question"], config)
        records.append(
            # ragas 0.4's column names. The old question/answer/contexts/ground_truth
            # set still produces numbers, which is the trap: context_precision needs
            # `reference` and silently scored against nothing under the old names,
            # reporting 0.12 where the retrieval was fine.
            {
                "user_input": item["question"],
                "response": answer,
                "retrieved_contexts": contexts if contexts else [""],
                "reference": item.get("ground_truth", ""),
            }
        )
        print(f"  answered: {item['question'][:60]}")

    print("Scoring with Ragas...")
    scores = _score_with_ragas(records)

    corpus_year = corpus_tax_year([c for r in records for c in r["retrieved_contexts"]])
    per_question = []
    for record, sample in zip(records, _per_sample_metrics(scores), strict=False):
        pinned = _pinned_year(record["reference"])
        per_question.append(
            {
                **record,
                "scores": sample,
                # the ground truth names a tax year the ingested revision is not
                # about, so a correct refusal scores as a failure
                "stale_ground_truth": bool(pinned and corpus_year and pinned != corpus_year),
            }
        )

    report = {
        "metrics": _aggregate_metrics(scores),
        "n_questions": len(items),
        "corpus_tax_year": corpus_year,
        "ground_truth_year_mismatch": [
            r["user_input"] for r in per_question if r.get("stale_ground_truth")
        ],
        "metrics_excluding_year_mismatch": _aggregate_metrics_for(
            scores, [i for i, r in enumerate(per_question) if not r.get("stale_ground_truth")]
        ),
        "per_question": per_question,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {report_path}")
    print("Scores:", report["metrics"])
    return report
