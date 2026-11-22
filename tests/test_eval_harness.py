from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from taxcite.eval_harness import _aggregate_metrics


class _FakeColumn:
    def __init__(self, values):
        self._values = values

    def mean(self):
        return sum(self._values) / len(self._values)


class _FakeFrame:
    """Stand-in for the DataFrame that EvaluationResult.to_pandas() returns."""

    def __init__(self, data):
        self._data = data

    @property
    def columns(self):
        return list(self._data)

    def __getitem__(self, key):
        return _FakeColumn(self._data[key])


class _FakeScores:
    def __init__(self, data):
        self._data = data

    def to_pandas(self):
        return _FakeFrame(self._data)


def test_aggregate_metrics_means_each_column():
    scores = _FakeScores(
        {
            "faithfulness": [1.0, 0.8],
            "answer_relevancy": [0.9, 0.7],
            "context_precision": [1.0, 1.0],
            "question": ["a", "b"],  # non-metric column is ignored
        }
    )
    result = _aggregate_metrics(scores)
    assert result == {
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "context_precision": 1.0,
    }


def test_aggregate_metrics_skips_missing_columns():
    scores = _FakeScores({"faithfulness": [0.5, 0.5]})
    assert _aggregate_metrics(scores) == {"faithfulness": 0.5}


def test_per_sample_metrics_keeps_each_question_score():
    """The report used to carry only column means, so comparing a retrieval change
    against a baseline could not be done as a paired test over the same questions."""
    pd = pytest.importorskip("pandas")  # arrives with the eval extra, not with dev

    from taxcite.eval_harness import _per_sample_metrics

    scores = MagicMock()
    scores.to_pandas.return_value = pd.DataFrame(
        {
            "faithfulness": [1.0, 0.5],
            "answer_relevancy": [0.8, 0.2],
            "context_precision": [0.9, 0.1],
        }
    )
    assert _per_sample_metrics(scores) == [
        {"faithfulness": 1.0, "answer_relevancy": 0.8, "context_precision": 0.9},
        {"faithfulness": 0.5, "answer_relevancy": 0.2, "context_precision": 0.1},
    ]


def test_per_sample_metrics_turns_nan_into_none():
    """Ragas leaves NaN where a metric could not be computed. json.dump writes that
    as bare NaN, which is not valid JSON and breaks anything reading the report."""
    pd = pytest.importorskip("pandas")  # arrives with the eval extra, not with dev

    from taxcite.eval_harness import _per_sample_metrics

    scores = MagicMock()
    scores.to_pandas.return_value = pd.DataFrame({"faithfulness": [float("nan")]})
    assert _per_sample_metrics(scores) == [{"faithfulness": None}]


def test_pinned_year_reads_the_tax_year_a_ground_truth_names():
    from taxcite.eval_harness import _pinned_year

    assert _pinned_year("For 2024, the standard deduction for a single filer is $14,600.") == "2024"
    assert _pinned_year("The 2025 tax year limit is higher.") == "2025"
    assert _pinned_year("Mortgage interest is deductible if the loan is secured.") is None


def test_corpus_tax_year_picks_the_year_the_corpus_is_about():
    from taxcite.eval_harness import corpus_tax_year

    chunks = ["for 2025 the limit is", "in 2025 taxpayers may", "2024 figures shown for comparison"]
    assert corpus_tax_year(chunks) == "2025"
    assert corpus_tax_year(["no years here at all"]) is None


def test_year_mismatch_is_what_makes_a_ground_truth_stale():
    """A dataset written against 2024 publications scores a correct refusal as a
    failure once the IRS serves the 2025 revision from the same URL."""
    from taxcite.eval_harness import _pinned_year, corpus_tax_year

    corpus = corpus_tax_year(["2025 " * 20, "2024"])
    assert corpus == "2025"
    pinned = _pinned_year("For 2024, the standard deduction for a single filer is $14,600.")
    assert pinned != corpus


class TestProvenance:
    """A published score with no model version and no revision cannot be compared
    against a later one: a difference could be the change you made or a model that
    moved under you. Resolving that ambiguity once cost a full baseline re-run."""

    def test_records_what_produced_the_numbers(self):
        from taxcite.eval_harness import _provenance

        p = _provenance()
        assert p["generated_at"].endswith("+00:00")
        assert p["generator_model"]
        assert p["judge_model"]
        # The corpus and the Ragas judge embed with different models; recording
        # one would misdescribe half the pipeline.
        assert p["corpus_embedding_model"] != p["judge_embedding_model"]
        assert isinstance(p["git_dirty"], bool)

    def test_honours_the_model_override(self, monkeypatch):
        from taxcite.eval_harness import _provenance

        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-5")
        assert _provenance()["generator_model"] == "claude-opus-5"

    def test_survives_a_missing_git(self, monkeypatch):
        """Reports still have to be writable outside a checkout, e.g. in a container."""
        import subprocess

        from taxcite import eval_harness

        def boom(*a, **k):
            raise FileNotFoundError("git")

        monkeypatch.setattr(subprocess, "run", boom)
        p = eval_harness._provenance()
        assert p["git_revision"] is None


class TestScoreCoverage:
    """Ragas returns NaN when a judge call fails, and pandas' mean skips NaN, so a
    run that lost most of its scores still reports a confident headline. One did:
    9/50, 14/50 and 7/50 scored, published as 0.9533 / 0.6766 / 0.9195, exit code
    0, no error logged. The surviving questions skewed high, so the broken run
    looked better than the healthy one it was compared against."""

    def test_counts_how_many_questions_actually_scored(self):
        from taxcite.eval_harness import _score_coverage

        per_question = [
            {"scores": {"faithfulness": 0.9, "context_precision": 1.0}},
            {"scores": {"faithfulness": None, "context_precision": 0.5}},
            {"scores": {"faithfulness": 0.8, "context_precision": None}},
        ]
        cov = _score_coverage(per_question)
        assert cov["faithfulness"] == {"scored": 2, "of": 3}
        assert cov["context_precision"] == {"scored": 2, "of": 3}

    def test_rejects_a_run_that_lost_most_of_its_scores(self):
        from taxcite.eval_harness import MIN_SCORE_COVERAGE, check_coverage

        per_question = [{"scores": {"faithfulness": 0.9 if i < 7 else None}} for i in range(50)]
        with pytest.raises(ValueError, match="faithfulness"):
            check_coverage(_cov(per_question))
        assert MIN_SCORE_COVERAGE > 0.5

    def test_accepts_a_run_that_lost_one_judge_call(self):
        """A single flaky judge reply should not throw away a 50-question run."""
        from taxcite.eval_harness import check_coverage

        per_question = [{"scores": {"faithfulness": None if i == 0 else 0.9}} for i in range(50)]
        check_coverage(_cov(per_question))


def _cov(per_question):
    from taxcite.eval_harness import _score_coverage

    return _score_coverage(per_question)


def _judge_response(usage_metadata=None, llm_output=None):
    """An LLMResult-shaped stand-in. SimpleNamespace rather than a class so the
    fixtures carry no mutable class attributes."""
    from types import SimpleNamespace

    generations = []
    if usage_metadata is not None:
        message = SimpleNamespace(usage_metadata=usage_metadata)
        generations = [[SimpleNamespace(message=message)]]
    return SimpleNamespace(generations=generations, llm_output=llm_output)


def test_judge_call_cost_prices_usage_metadata():
    from taxcite import cost
    from taxcite.eval_harness import judge_call_cost

    response = _judge_response(usage_metadata={"input_tokens": 1000, "output_tokens": 200})
    expected = (
        1000 * cost.ANTHROPIC_INPUT_COST_PER_TOKEN
        + 200 * cost.ANTHROPIC_OUTPUT_COST_PER_TOKEN
    )
    assert judge_call_cost(response) == expected


def test_judge_call_cost_falls_back_to_llm_output_and_legacy_names():
    from taxcite import cost
    from taxcite.eval_harness import judge_call_cost

    response = _judge_response(
        llm_output={"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    )
    expected = (
        10 * cost.ANTHROPIC_INPUT_COST_PER_TOKEN
        + 5 * cost.ANTHROPIC_OUTPUT_COST_PER_TOKEN
    )
    assert judge_call_cost(response) == expected


def test_judge_call_cost_raises_rather_than_pricing_an_unknown_shape():
    import pytest

    from taxcite import cost
    from taxcite.eval_harness import judge_call_cost

    with pytest.raises(cost.CostBudgetExceeded, match="no token usage"):
        judge_call_cost(_judge_response(llm_output={}))

    with pytest.raises(cost.CostBudgetExceeded, match="no recognised token counts"):
        judge_call_cost(_judge_response(llm_output={"usage": {"tokens_in": 1, "tokens_out": 2}}))


def _llm_result(reason_key=None, reason=None, generation_info=None):
    """An LLMResult-shaped stand-in carrying one generation."""
    from types import SimpleNamespace

    metadata = {reason_key: reason} if reason_key else {}
    generation = SimpleNamespace(
        generation_info=generation_info,
        message=SimpleNamespace(response_metadata=metadata),
    )
    flat = SimpleNamespace(generations=[[generation]])
    return SimpleNamespace(flatten=lambda: [flat])


def test_judge_finished_accepts_anthropics_end_turn():
    from taxcite.eval_harness import judge_finished

    # langchain-anthropic 1.7 sets finish_reason to Anthropic's own value, which
    # ragas 0.4 checks against OpenAI's list. This is the regression.
    assert judge_finished(_llm_result("finish_reason", "end_turn")) is True
    assert judge_finished(_llm_result("stop_reason", "end_turn")) is True


def test_judge_finished_still_rejects_a_truncated_response():
    from taxcite.eval_harness import judge_finished

    # The check exists for this case, so widening the vocabulary must not swallow it.
    assert judge_finished(_llm_result("finish_reason", "max_tokens")) is False
    assert judge_finished(_llm_result("stop_reason", "max_tokens")) is False


def test_judge_finished_keeps_the_openai_and_vertex_vocabulary():
    from taxcite.eval_harness import judge_finished

    for reason in ("stop", "STOP", "MAX_TOKENS", "eos_token"):
        assert judge_finished(_llm_result("finish_reason", reason)) is True


def test_judge_finished_defaults_to_finished_when_nothing_reports_a_reason():
    from taxcite.eval_harness import judge_finished

    assert judge_finished(_llm_result()) is True


def test_judge_finished_prefers_generation_info_when_present():
    from taxcite.eval_harness import judge_finished

    truncated = _llm_result("finish_reason", "end_turn", generation_info={"finish_reason": "max_tokens"})
    assert judge_finished(truncated) is False


def test_judge_cost_cap_raises_rather_than_logging():
    """langchain_core defaults BaseCallbackHandler.raise_error to False and only
    re-raises when it is True, so a cap that does not set it is a log line
    rather than a ceiling. Imported directly: _judge_llm needs langchain_anthropic
    from the eval extra, which the default CI sync does not install."""
    import pytest
    from langchain_core.callbacks.manager import handle_event

    from taxcite.eval_harness import JudgeCostCap

    assert JudgeCostCap.raise_error is True

    class _Boom(JudgeCostCap):
        def on_llm_end(self, response, **_kwargs):
            raise RuntimeError("cap tripped")

    with pytest.raises(RuntimeError, match="cap tripped"):
        handle_event([_Boom()], "on_llm_end", None, None)
