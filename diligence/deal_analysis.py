"""Derives per-deal unit-economics and fund-return summaries for the memo.

This is the integration glue described in Module 4: turning a single
shortlisted deal's own synthetic fields into inputs for Module 2 (cost
curve) and Module 3 (dilution/returns), so the IC memo can report
deal-specific numbers rather than a single canonical example. All
per-deal cost-curve assumptions are drawn from a seeded distribution
derived from the deal id -- see :func:`diligence.checklist.deal_seed`.
"""

from __future__ import annotations

import math

import numpy as np

from diligence.checklist import deal_seed
from diligence.memo import FundReturnSummary, UnitEconomicsSummary
from fund_returns.dilution import FinancingRound, run_exit_scenarios, simulate_ownership
from screening.data_generator import Deal
from unit_economics.learning_curve import CostCurveParams, years_to_margin_threshold
from unit_economics.monte_carlo import MonteCarloConfig, TriangularRange, run_monte_carlo

_MAX_YEARS = 25
_TARGET_MARGIN = 0.30
_DEMO_PRICE_PER_UNIT = 3500.0
_DEMO_V0 = 100.0
_MC_PATHS_PER_DEAL = 2000


def derive_unit_economics_summary(deal: Deal, seed: int = 42) -> UnitEconomicsSummary:
    """Deal-specific cost-curve outlook, or a note explaining why the
    hardware learning-curve model does not apply to this deal's sector."""
    if not deal.is_hardware_sector:
        return UnitEconomicsSummary(
            applicable=False,
            note=(
                "This toolkit's Wright's Law cost-curve model is a hardware deployment-volume "
                "proxy; it is not applied to non-hardware sectors like this deal's."
            ),
        )

    rng = np.random.default_rng(deal_seed(seed, deal.deal_id))
    starting_cost = float(rng.uniform(6000.0, 13000.0))
    growth_rate = float(rng.uniform(0.15, 0.45))
    learning_rate = float(rng.uniform(0.78, 0.92))

    params = CostCurveParams(
        starting_cost_per_unit=starting_cost,
        v0=_DEMO_V0,
        growth_rate=growth_rate,
        learning_rate=learning_rate,
        price_per_unit=_DEMO_PRICE_PER_UNIT,
    )
    breakeven_year = years_to_margin_threshold(params, target_margin=0.0, max_years=_MAX_YEARS)
    target_year = years_to_margin_threshold(
        params, target_margin=_TARGET_MARGIN, max_years=_MAX_YEARS
    )

    mc_config = MonteCarloConfig(
        n_paths=_MC_PATHS_PER_DEAL,
        seed=deal_seed(seed, deal.deal_id),
        v0=_DEMO_V0,
        price_per_unit=_DEMO_PRICE_PER_UNIT,
        max_years=_MAX_YEARS,
        starting_cost_per_unit=TriangularRange(
            starting_cost * 0.85, starting_cost, starting_cost * 1.15
        ),
        growth_rate=TriangularRange(growth_rate * 0.7, growth_rate, min(growth_rate * 1.3, 0.9)),
        learning_rate=TriangularRange(
            max(learning_rate - 0.08, 0.5), learning_rate, min(learning_rate + 0.08, 0.99)
        ),
    )
    mc_result = run_monte_carlo(mc_config)
    median_years = mc_result.percentiles.get(50, math.nan)

    return UnitEconomicsSummary(
        applicable=True,
        breakeven_year=breakeven_year,
        target_margin_year=target_year,
        mc_median_years=None if math.isnan(median_years) else median_years,
        mc_pct_never_profitable=mc_result.pct_never_profitable,
        note="Deal-specific synthetic cost-curve assumptions; see Module 2 for the canonical demo.",
    )


def derive_fund_return_summary(
    deal: Deal, investment_usd: float | None = None
) -> FundReturnSummary:
    """Ownership/return summary using round sizes scaled off this deal's
    own ask and pre-money valuation, and exit values scaled off its TAM.

    Simplifying assumption: round progression is a fixed multiple of the
    entry round's size, and the fund always takes a 50%-of-entry-check
    follow-on at Series B. Real follow-on decisions depend on company
    performance, not a formula -- see the README.
    """
    investment = investment_usd if investment_usd is not None else deal.ask_amount_usd

    rounds = [
        FinancingRound(
            name="Series B",
            pre_money_usd=deal.pre_money_valuation_usd * 4.0,
            raise_amount_usd=deal.ask_amount_usd * 1.5,
            fund_participates=True,
            fund_follow_on_usd=investment * 0.5,
        ),
        FinancingRound(
            name="Series C",
            pre_money_usd=deal.pre_money_valuation_usd * 12.0,
            raise_amount_usd=deal.ask_amount_usd * 3.0,
        ),
        FinancingRound(
            name="Series D",
            pre_money_usd=deal.pre_money_valuation_usd * 30.0,
            raise_amount_usd=deal.ask_amount_usd * 6.0,
        ),
    ]

    snapshots = simulate_ownership(
        investment, deal.pre_money_valuation_usd, deal.ask_amount_usd, rounds
    )
    initial, final = snapshots[0], snapshots[-1]

    exit_scenarios = {
        "Write-off": 0.0,
        "Modest exit": 0.01 * deal.tam_usd,
        "Good exit": 0.05 * deal.tam_usd,
        "Home run": 0.15 * deal.tam_usd,
    }
    outcomes = run_exit_scenarios(
        final.ownership_pct, final.cumulative_invested_usd, exit_scenarios
    )

    return FundReturnSummary(
        initial_ownership_pct=initial.ownership_pct,
        final_ownership_pct=final.ownership_pct,
        total_invested_usd=final.cumulative_invested_usd,
        exit_outcomes=tuple(outcomes),
    )
