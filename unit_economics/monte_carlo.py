"""Monte Carlo simulation over uncertain learning-rate and deployment-growth
parameters to produce a distribution of "years to profitability" rather than
a single point estimate.

Each simulated path independently samples a learning rate, a deployment
growth rate, and a starting unit cost from triangular distributions (a
common choice for expert-elicited ranges: it needs only a low/mode/high and
concentrates mass near the most-likely value). All paths share the same
price and starting deployment volume so that variance in the outcome can be
attributed to the sampled parameters rather than fixed inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

PERCENTILES: tuple[int, ...] = (10, 25, 50, 75, 90)


@dataclass(frozen=True)
class TriangularRange:
    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        if not (self.low <= self.mode <= self.high):
            raise ValueError("TriangularRange requires low <= mode <= high")


@dataclass(frozen=True)
class MonteCarloConfig:
    n_paths: int = 5000
    seed: int = 123
    v0: float = 100.0
    price_per_unit: float = 3500.0
    target_margin: float = 0.0
    max_years: int = 25
    learning_rate: TriangularRange = field(
        default_factory=lambda: TriangularRange(0.72, 0.85, 0.95)
    )
    growth_rate: TriangularRange = field(
        default_factory=lambda: TriangularRange(0.10, 0.30, 0.55)
    )
    starting_cost_per_unit: TriangularRange = field(
        default_factory=lambda: TriangularRange(6000.0, 9000.0, 13000.0)
    )


@dataclass(frozen=True)
class MonteCarloResult:
    config: MonteCarloConfig
    years_to_profit: np.ndarray  # shape (n_paths,), -1 means censored (never reached)
    percentiles: dict[int, float]  # over uncensored paths only
    mean_years: float  # over uncensored paths only
    pct_never_profitable: float


def _sample_triangular(
    rng: np.random.Generator, rng_spec: TriangularRange, size: int
) -> np.ndarray:
    return rng.triangular(rng_spec.low, rng_spec.mode, rng_spec.high, size=size)


def run_monte_carlo(config: MonteCarloConfig | None = None) -> MonteCarloResult:
    cfg = config or MonteCarloConfig()
    rng = np.random.default_rng(cfg.seed)

    learning_rates = _sample_triangular(rng, cfg.learning_rate, cfg.n_paths)
    growth_rates = _sample_triangular(rng, cfg.growth_rate, cfg.n_paths)
    starting_costs = _sample_triangular(rng, cfg.starting_cost_per_unit, cfg.n_paths)

    b = np.log2(learning_rates)  # negative exponents, one per path

    years_to_profit = np.full(cfg.n_paths, -1, dtype=int)
    found = np.zeros(cfg.n_paths, dtype=bool)

    for year in range(cfg.max_years + 1):
        volume = cfg.v0 * (1.0 + growth_rates) ** year
        cost = starting_costs * (volume / cfg.v0) ** b
        margin = (cfg.price_per_unit - cost) / cfg.price_per_unit
        newly_found = (~found) & (margin >= cfg.target_margin)
        years_to_profit[newly_found] = year
        found |= newly_found
        if found.all():
            break

    uncensored = years_to_profit[years_to_profit >= 0]
    pct_never_profitable = 1.0 - (len(uncensored) / cfg.n_paths)

    if len(uncensored) > 0:
        percentiles = {p: float(np.percentile(uncensored, p)) for p in PERCENTILES}
        mean_years = float(np.mean(uncensored))
    else:
        percentiles = dict.fromkeys(PERCENTILES, math.nan)
        mean_years = math.nan

    return MonteCarloResult(
        config=cfg,
        years_to_profit=years_to_profit,
        percentiles=percentiles,
        mean_years=mean_years,
        pct_never_profitable=pct_never_profitable,
    )
