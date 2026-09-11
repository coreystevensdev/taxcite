"""Rolling-cost-cap singleton and pricing constants shared across embed and agent calls."""
from __future__ import annotations

from rolling_cost_cap import CostCap

# Voyage AI voyage-3.5-lite: $0.02 per million tokens
VOYAGE_COST_PER_TOKEN: float = 0.02 / 1_000_000

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

generation_cap = CostCap(
    multiplier=3.0,
    window=50,
    min_samples=5,
    absolute_ceiling=0.10,
    monthly_budget=8.00,
)


class CostBudgetExceeded(RuntimeError):
    pass


class AnswerTruncated(RuntimeError):
    """The model ran out of tokens before completing its submit_answer call."""
