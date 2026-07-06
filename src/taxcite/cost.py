"""Rolling-cost-cap singleton and pricing constants shared across embed and agent calls."""
from __future__ import annotations

from rolling_cost_cap import CostCap

# Voyage AI voyage-3.5-lite: $0.02 per million tokens
VOYAGE_COST_PER_TOKEN: float = 0.02 / 1_000_000

# Anthropic claude-sonnet-4-6: $3/M input tokens, $15/M output tokens
ANTHROPIC_INPUT_COST_PER_TOKEN: float = 3.0 / 1_000_000
ANTHROPIC_OUTPUT_COST_PER_TOKEN: float = 15.0 / 1_000_000

cap = CostCap(
    multiplier=3.0,
    window=50,
    min_samples=5,
    absolute_ceiling=0.10,
    monthly_budget=10.00,
)


class CostBudgetExceeded(RuntimeError):
    pass
