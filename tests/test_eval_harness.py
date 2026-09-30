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
