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

generation_cap = CostCap(
    multiplier=3.0,
    window=50,
    min_samples=5,
    absolute_ceiling=0.10,
    monthly_budget=8.00,
)


# The Ragas judge, which was the only uncapped spender until 2026-11-21.
#
# multiplier=8.0, not the 3.0 the three above use, and the reason is the same
# one this file gives for splitting their windows: a rolling median only
# calibrates to a homogeneous workload. The judge is not one. It runs a
# different prompt per metric, and context_precision judges a single retrieved
# chunk while faithfulness judges a whole answer, so one window mixes call
# shapes that differ several-fold.
#
# Measured on the 2026-11-21 run rather than guessed: observed judge costs
# ranged $0.0138 to $0.0621, a 4.5x spread, and at 3.0 the cap tripped 292
# times and killed more than half the calls, leaving faithfulness at 22 of 50
# and the run unpublishable. Every one of those trips was the median layer;
# neither fixed layer fired. fairline's generation cap is 8.0 for the same
# arithmetic, with the same 4.5x figure in its own comment.
#
# The two fixed layers are what actually bound spend. That run reached $13.57
# of judge spend before failing, so a completed run costs more than that and
# the monthly budget stays at 25 until a full one reports a number.
judge_cap = CostCap(
    multiplier=8.0,
    window=50,
    min_samples=5,
    absolute_ceiling=0.25,
    monthly_budget=25.00,
)


class CostBudgetExceeded(RuntimeError):
    pass


class AnswerTruncated(RuntimeError):
    """The model ran out of tokens before completing its submit_answer call."""
