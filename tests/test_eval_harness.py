from __future__ import annotations

from unittest.mock import MagicMock

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
    import pandas as pd

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
    import pandas as pd

    from taxcite.eval_harness import _per_sample_metrics

    scores = MagicMock()
    scores.to_pandas.return_value = pd.DataFrame({"faithfulness": [float("nan")]})
    assert _per_sample_metrics(scores) == [{"faithfulness": None}]
