"""Ragas eval harness for TaxCite.

Runs the agent on a JSONL dataset, scores with Ragas faithfulness /
answer_relevancy / context_precision, and writes a JSON report.

Usage:
    python -m taxcite eval [--dataset eval/dataset.jsonl] [--out eval/report.json]

Requires the eval extras (ragas, datasets) and a live DATABASE_URL with
ingested publications. The judge is Claude and the judge's embedder is
voyage-3, both wired explicitly in `_judge_llm` and `_judge_embeddings`, so
scoring needs no provider beyond the two the agent already uses. Neither
OPENAI_API_KEY nor RAGAS_LLM is read anywhere in this project.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

from taxcite import cost
from taxcite.embed import EMBED_MODEL

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


def judge_call_cost(response) -> float:
    """Dollar cost of one judge response.

    Ragas drives the judge through LangChain, so usage lives on the message
    rather than on an Anthropic response object, and the key names differ by
    version. Raises when it finds none: an unpriced call is an uncapped one,
    which is the thing this exists to stop.
    """
    usage = None
    for generation_list in getattr(response, "generations", None) or []:
        for generation in generation_list:
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if usage:
                break
        if usage:
            break
    if not usage:
        usage = (getattr(response, "llm_output", None) or {}).get("usage")
    if not usage:
        raise cost.CostBudgetExceeded(
            "judge response carried no token usage, so its cost cannot be bounded"
        )

    def read(*names):
        return next((usage[n] for n in names if n in usage), None)

    sent = read("input_tokens", "prompt_tokens")
    produced = read("output_tokens", "completion_tokens")
    if sent is None or produced is None:
        raise cost.CostBudgetExceeded(
            f"judge usage has no recognised token counts: {sorted(usage)}"
        )
    return (
        sent * cost.ANTHROPIC_INPUT_COST_PER_TOKEN
        + produced * cost.ANTHROPIC_OUTPUT_COST_PER_TOKEN
    )


def judge_finished(response) -> bool:
    """Whether every generation in a judge response actually completed.

    ragas 0.4's default parser reads `response_metadata["finish_reason"]` and
    checks it against OpenAI's vocabulary, ["stop", "STOP", "MAX_TOKENS",
    "eos_token"]. Its Anthropic branch, which does accept "end_turn", is an
    `elif` reached only when finish_reason is absent. langchain-anthropic 1.7
    sets both keys to Anthropic's own value, so the OpenAI check runs, fails,
    and the Anthropic branch never executes. Every judge call then raised
    LLMDidNotFinishException("...increase the max_tokens...") and a whole run
    scored 0 of 50 on all three metrics, which is what happened on the first
    armed run.

    "max_tokens", Anthropic's genuinely-truncated reason, is deliberately not
    accepted. That is the case the check exists for.
    """
    finished = {"stop", "STOP", "MAX_TOKENS", "eos_token", "end_turn"}
    for flat in response.flatten():
        generation = flat.generations[0][0]
        info = generation.generation_info or {}
        metadata = getattr(getattr(generation, "message", None), "response_metadata", None) or {}
        reason = (
            info.get("finish_reason")
            or metadata.get("finish_reason")
            or metadata.get("stop_reason")
        )
        if reason is not None and reason not in finished:
            return False
    return True


logger = logging.getLogger(__name__)


class JudgeCostCap(BaseCallbackHandler):
    """Prices each judge call and stops the run when the cap trips.

    raise_error matters more than it looks. langchain_core defaults it to False
    (callbacks/base.py) and only re-raises when it is True (manager.py), so
    without it a raising on_llm_end is logged and execution continues: the cap
    becomes a log line rather than a ceiling. The first failed run showed
    exactly that, twelve trips and a run that finished anyway.
    """

    raise_error = True

    def on_llm_end(self, response, **_kwargs) -> None:
        decision = cost.judge_cap.evaluate(judge_call_cost(response))
        # Every call, not only the blocked ones. The 2026-11-22 run recorded 24
        # trips and nothing else, so the distribution they were outliers against
        # had to be inferred, and the inference did not survive arithmetic.
        logger.info(
            "judge call $%.6f (median $%s, monthly $%.4f)",
            decision.observed,
            f"{decision.median:.6f}" if decision.median is not None else "none",
            decision.monthly_spend,
        )
        if not decision.allowed:
            raise cost.CostBudgetExceeded(
                f"judge blocked by cost cap ({decision.trip}): "
                f"${decision.observed:.6f} observed, "
                f"${decision.monthly_spend:.4f} monthly"
            )


def _judge_llm():
    """Claude as the judge, rather than whatever Ragas reaches for by default.

    Left unset, `evaluate` builds an OpenAI client and fails on auth, which is a
    third provider this project neither configures nor needs. That failure is the
    whole reason the judge is passed in explicitly.
    """
    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        is_finished_parser=judge_finished,
        langchain_llm=ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            temperature=0,
            max_tokens=8192,
            callbacks=[JudgeCostCap()],
        )
    )


def _judge_embeddings():
    """answer_relevancy embeds the generated question, so Ragas needs embeddings
    of its own. Voyage again, so the judge measures distance the same way
    retrieval does."""
    from langchain_voyageai import VoyageAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(VoyageAIEmbeddings(model="voyage-3"))


# A ground truth that opens "For 2024, ..." is pinned to a tax year, and the
# corpus is pinned separately by manifest.TAX_YEAR. Move one without the other
# and the agent correctly refuses to state the year it does not have, which
# scores as a failure. This is what tells those apart from real ones.
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


# A judge call that fails leaves NaN, and pandas' mean skips NaN, so a run that
# lost most of its scores still produces a confident headline. One did: 7 of 50
# context_precision scores survived and the mean read 0.9195, higher than the
# healthy run it was being compared against, because the questions that dropped
# out were the hard ones. Losing a single flaky judge reply is tolerable; losing
# the dataset is not a number worth publishing.
MIN_SCORE_COVERAGE = 0.9


def _score_coverage(per_question: list[dict]) -> dict[str, dict[str, int]]:
    """How many questions actually produced a score, per metric."""
    total = len(per_question)
    coverage = {}
    for name in METRIC_NAMES:
        scored = sum(
            1
            for q in per_question
            if (q.get("scores") or {}).get(name) is not None
        )
        if scored or any(name in (q.get("scores") or {}) for q in per_question):
            coverage[name] = {"scored": scored, "of": total}
    return coverage


def check_coverage(coverage: dict[str, dict[str, int]]) -> None:
    """Refuse to treat a run as publishable when the judges mostly did not answer."""
    thin = [
        f"{name} scored {c['scored']}/{c['of']}"
        for name, c in coverage.items()
        if c["of"] and c["scored"] / c["of"] < MIN_SCORE_COVERAGE
    ]
    if thin:
        raise ValueError(
            "eval run is not publishable, the judges did not score enough questions: "
            + "; ".join(thin)
            + f" (need {MIN_SCORE_COVERAGE:.0%}). Re-run; this is usually transient."
        )


def _provenance() -> dict:
    """What produced these numbers.

    A published score with no model version and no code revision cannot be
    compared against anything later: a difference could be the retrieval change
    you made or a model that moved under you, and nothing in the file tells you
    which. That ambiguity cost a full re-run of the baseline to resolve once.
    """
    import subprocess

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=5, check=True
            )
            return out.stdout.strip() or None
        except (subprocess.SubprocessError, OSError):
            return None

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator_model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        "judge_model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        # The corpus and the Ragas judge do not use the same embedding model,
        # so recording one number would misdescribe half the pipeline.
        "corpus_embedding_model": EMBED_MODEL,
        "judge_embedding_model": "voyage-3",
        "git_revision": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    }


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
        "provenance": _provenance(),
        "score_coverage": _score_coverage(per_question),
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

    # Before writing, not after: a thin run must not overwrite a good report.
    check_coverage(report["score_coverage"])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {report_path}")
    print("Scores:", report["metrics"])
    return report
