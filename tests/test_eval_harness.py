from __future__ import annotations

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
