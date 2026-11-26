"""Rolling-cost-cap singleton and pricing constants shared across embed and agent calls."""
from __future__ import annotations

from rolling_cost_cap import CostCap

# Voyage AI voyage-3.5-lite: $0.02 per million tokens
VOYAGE_COST_PER_TOKEN: float = 0.02 / 1_000_000
VOYAGE_RERANK_COST_PER_TOKEN: float = 0.05 / 1_000_000

# Anthropic claude-sonnet-4-6: $3/M input tokens, $15/M output tokens
ANTHROPIC_INPUT_COST_PER_TOKEN: float = 3.0 / 1_000_000
ANTHROPIC_OUTPUT_COST_PER_TOKEN: float = 15.0 / 1_000_000

# One window per workload. Embedding and generation differ by roughly fifty times
# per token, so a shared rolling median is not measuring either of them: after an
# ingest the window fills with Voyage calls, the median collapses toward those, and
# every Claude call reads as an anomaly at 3x. That made `taxcite eval` impossible
# to finish, since each question embeds its query before generating, and by the
# fifth sample the median is an embed cost.
#
# Budgets split 2/8 of the old $10: embedding a full corpus is cents, generation is
# where the money goes.
embed_cap = CostCap(
    multiplier=3.0,
    window=50,
    min_samples=5,
    absolute_ceiling=0.01,
    monthly_budget=2.00,
)

# Its own window rather than sharing the embed one. Reranking costs about 2.5x
# per token and sees a whole candidate set per call instead of one query, so the
# two have different enough distributions that a shared rolling median would
# describe neither. That is the same mistake that made generation unrunnable
# after an ingest.
rerank_cap = CostCap(
    multiplier=3.0,
    window=50,
    min_samples=5,
    absolute_ceiling=0.02,
    monthly_budget=2.00,
)

# embed_texts batches up to 128 chunks; embed_query embeds one question. The two
# costs differ by about three orders of magnitude, so a shared rolling median
# describes neither, which is the same reason rerank_cap has its own window.
#
# Sharing it did more than blur the median, it killed the public demo. In steady
# state the window fills with query costs and the median settles near
# $0.00000015 (an 8-token suggested question). A visitor then types a 34-token
# question costing $0.00000068, which is 4.5x the median, trips the 3x
# multiplier, and returns 503. Total spend at that point was $0.0000024 against
# a $2.00 budget, so nothing was overspent; the anomaly detector was reading
# ordinary question length as an anomaly.
#
# The rolling-median layer is deliberately inert here. Question length varies by
# an order of magnitude between legitimate questions, so any multiplier tight
# enough to catch a real spike is too tight for this workload. The absolute
# ceiling is the guard that means something: $0.0005 is 25,000 tokens, which no
# question is, and a pasted document hits it immediately.
query_embed_cap = CostCap(
    multiplier=1_000.0,
    window=50,
    min_samples=5,
    absolute_ceiling=0.0005,
    monthly_budget=1.00,
)

generation_cap = CostCap(
    multiplier=3.0,
    window=50,
    min_samples=5,
    absolute_ceiling=0.10,
    monthly_budget=8.00,
)


# The Ragas judge, which was the only uncapped spender until 2026-11-21.
#
# The median layer is deliberately inert here, and multiplier=50 is how that is
# spelled: CostCap has no off switch, so the number is chosen to sit above the
# $0.25 per-call ceiling for any median this workload produces. At a median of
# $0.006, the lowest the 2026-11-22 run implies, 50x is $0.30 and the ceiling
# governs instead. It comes back to life only if the median falls under $0.005.
#
# Turning it off rather than tuning it, because this file's own argument for
# giving embed, rerank and generation separate windows applies to the judge and
# says the median cannot work here: it runs a different prompt per metric, and
# context_precision judges a single retrieved chunk where faithfulness judges a
# whole answer.
#
# Two runs measured that, rather than it being an argument from design. At 3.0
# the cap tripped 292 times and left faithfulness at 22 of 50. At 8.0 it tripped
# 24 times, every one a faithfulness-sized call between $0.0493 and $0.0618, and
# left faithfulness at 45 of 50. For a $0.055 call to trip at 8x, the window
# median has to be under $0.007, so the window is dominated by the cheap metric
# and the expensive one reads as an anomaly against it. That gap is structural,
# so no multiplier separates anomaly from metric: anything admitting faithfulness
# is already past 8, and the cheap-call median drags the bar back under it.
#
# 45 of 50 is publishable (MIN_SCORE_COVERAGE is 0.9) and not promotable:
# tests/test_eval_page.py holds the committed report to every question scored.
#
# The two fixed layers are what bound this workload, and the ceiling is already
# 4x the largest judge call ever observed. A complete run costs about $6.22.
judge_cap = CostCap(
    multiplier=50.0,
    window=50,
    min_samples=5,
    absolute_ceiling=0.25,
    monthly_budget=25.00,
)


class CostBudgetExceeded(RuntimeError):
    pass


class AnswerTruncated(RuntimeError):
    """The model ran out of tokens before completing its submit_answer call."""
