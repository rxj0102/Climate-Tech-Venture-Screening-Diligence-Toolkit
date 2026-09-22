"""Power-law portfolio construction simulator.

This is the well-known back-of-envelope VC exercise: a small share of
portfolio bets ("winners") return a large multiple, and the rest mostly
return less than 1x. Given a fund size and average check size, how many
bets does the fund make, and what fraction of those bets need to be
winners to return the fund (or hit a target MOIC)?

Simplifying assumptions (documented, see README):
- Every bet is the same check size (no reserves, no differentiated sizing).
- "Winner" vs. "not winner" is a single Bernoulli draw per bet; there is no
  correlation between bets (no sector or vintage clustering).
- Winner outcomes are drawn from a Lomax (shifted Pareto) distribution,
  which has a genuine heavy right tail; non-winners are drawn from a
  bounded uniform distribution representing write-offs through modest
  returns. Real fund outcomes are messier and rounds/dilution (Module 3's
  dilution model) are ignored here entirely -- this is a separate, simpler
  "portfolio construction" lens, not a reconciliation of the two models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PowerLawConfig:
    fund_size_usd: float
    avg_check_size_usd: float
    winner_alpha: float = 1.2  # Pareto shape; lower = heavier tail
    winner_scale: float = 10.0  # minimum multiple a "winner" can return
    loser_high_multiple: float = 1.5  # non-winners return U(0, this)
    n_sims: int = 5000
    seed: int = 2024

    def __post_init__(self) -> None:
        if self.fund_size_usd <= 0 or self.avg_check_size_usd <= 0:
            raise ValueError("fund_size_usd and avg_check_size_usd must be positive")
        if self.avg_check_size_usd > self.fund_size_usd:
            raise ValueError("avg_check_size_usd cannot exceed fund_size_usd")
        if self.winner_alpha <= 0:
            raise ValueError("winner_alpha must be positive")

    @property
    def num_bets(self) -> int:
        return int(self.fund_size_usd // self.avg_check_size_usd)


@dataclass(frozen=True)
class PortfolioSimulationResult:
    config: PowerLawConfig
    hit_rate: float
    fund_moics: np.ndarray  # one MOIC per simulated fund, shape (n_sims,)
    percentiles: dict[int, float]
    mean_moic: float


def _simulate_fund_moics(
    config: PowerLawConfig, hit_rate: float, common_uniforms: np.ndarray | None = None
) -> np.ndarray:
    """Simulate ``config.n_sims`` fund outcomes at a given ``hit_rate``.

    If ``common_uniforms`` is supplied (shape ``(n_sims, num_bets)``), it is
    used to decide which bets are winners instead of drawing fresh
    randomness, which enables the common-random-numbers technique used by
    :func:`required_hit_rate_for_target_moic` to guarantee monotonicity.
    """
    if not (0.0 <= hit_rate <= 1.0):
        raise ValueError("hit_rate must be between 0 and 1")

    rng = np.random.default_rng(config.seed)
    shape = (config.n_sims, config.num_bets)

    uniforms = common_uniforms if common_uniforms is not None else rng.random(shape)
    winner_multipliers = (rng.pareto(config.winner_alpha, size=shape) + 1.0) * config.winner_scale
    loser_multipliers = rng.uniform(0.0, config.loser_high_multiple, size=shape)

    is_winner = uniforms < hit_rate
    multipliers = np.where(is_winner, winner_multipliers, loser_multipliers)
    return np.asarray(multipliers.mean(axis=1))


def simulate_portfolio(config: PowerLawConfig, hit_rate: float) -> PortfolioSimulationResult:
    """Simulate the fund-level MOIC distribution at a given assumed hit rate."""
    fund_moics = _simulate_fund_moics(config, hit_rate)
    percentiles = {p: float(np.percentile(fund_moics, p)) for p in (10, 25, 50, 75, 90)}
    return PortfolioSimulationResult(
        config=config,
        hit_rate=hit_rate,
        fund_moics=fund_moics,
        percentiles=percentiles,
        mean_moic=float(np.mean(fund_moics)),
    )


def required_hit_rate_for_target_moic(
    config: PowerLawConfig,
    target_moic: float,
    hit_rate_grid: np.ndarray | None = None,
) -> tuple[float, float] | None:
    """Smallest hit rate (from ``hit_rate_grid``) whose simulated median
    fund MOIC meets or exceeds ``target_moic``.

    Uses a common-random-numbers technique: the same underlying uniform
    draws decide which bets are winners at every grid point, so raising the
    hit rate can only convert non-winners into winners and never the
    reverse. That makes the simulated median MOIC monotonically
    non-decreasing in the hit rate by construction, so the first grid point
    clearing the target is the answer -- no search algorithm needed.

    Returns ``(hit_rate, median_moic)``, or ``None`` if no grid point
    reaches the target.
    """
    grid = hit_rate_grid if hit_rate_grid is not None else np.arange(0.01, 0.51, 0.01)
    rng = np.random.default_rng(config.seed)
    shape = (config.n_sims, config.num_bets)
    common_uniforms = rng.random(shape)

    for hit_rate in grid:
        fund_moics = _simulate_fund_moics(config, float(hit_rate), common_uniforms=common_uniforms)
        median_moic = float(np.median(fund_moics))
        if median_moic >= target_moic:
            return float(hit_rate), median_moic
    return None
