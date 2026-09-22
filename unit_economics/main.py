"""Standalone entry point for Module 2: unit economics & cost-curve modeling.

Run with ``python -m unit_economics.main``. Models one synthetic
hardware-adjacent climate technology (a long-duration storage system) end
to end: deterministic path-to-profitability, a 5,000-path Monte Carlo over
uncertain assumptions, and a tornado sensitivity ranking.
"""

from __future__ import annotations

from unit_economics.learning_curve import CostCurveParams, years_to_margin_threshold
from unit_economics.monte_carlo import MonteCarloConfig, run_monte_carlo
from unit_economics.sensitivity import ParameterRange, tornado_sensitivity

BASE_PARAMS = CostCurveParams(
    starting_cost_per_unit=9000.0,
    v0=100.0,
    growth_rate=0.30,
    learning_rate=0.85,
    price_per_unit=3500.0,
)


def run() -> None:
    print("=" * 88)
    print("MODULE 2 -- UNIT ECONOMICS & TECHNOLOGY COST-CURVE MODEL")
    print("=" * 88)

    print("\nBase case (deterministic):")
    print(f"  {BASE_PARAMS}")
    breakeven_year = years_to_margin_threshold(BASE_PARAMS, target_margin=0.0, max_years=25)
    target_year = years_to_margin_threshold(BASE_PARAMS, target_margin=0.30, max_years=25)
    print(f"  Year gross margin crosses 0%:   {breakeven_year}")
    print(f"  Year gross margin reaches 30%:  {target_year}")

    print("\nMonte Carlo simulation (5,000 paths) over learning rate, growth rate, and cost:")
    mc_config = MonteCarloConfig(n_paths=5000, seed=123)
    result = run_monte_carlo(mc_config)
    print(f"  Paths simulated: {mc_config.n_paths}")
    print(
        f"  Pct never profitable within {mc_config.max_years} years: "
        f"{result.pct_never_profitable:.1%}"
    )
    print(f"  Mean years to profitability (uncensored paths): {result.mean_years:.2f}")
    print("  Percentiles (years to profitability, uncensored paths):")
    for p, v in result.percentiles.items():
        print(f"    p{p}: {v:.2f}")

    print("\nTornado sensitivity (years to profitability, one-at-a-time sweep):")
    parameter_ranges = {
        "learning_rate": ParameterRange(low=0.72, high=0.95),
        "growth_rate": ParameterRange(low=0.10, high=0.55),
        "starting_cost_per_unit": ParameterRange(low=6000.0, high=13000.0),
    }
    tornado = tornado_sensitivity(BASE_PARAMS, parameter_ranges, target_margin=0.0, max_years=25)
    for rank, t in enumerate(tornado, start=1):
        low_label = _format_years_label(t.years_at_low, t.reached_at_low)
        high_label = _format_years_label(t.years_at_high, t.reached_at_high)
        print(
            f"  {rank}. {t.parameter:<18} swing={t.swing:>5.1f}y  "
            f"low={low_label:<22} high={high_label}"
        )


def _format_years_label(years: float, reached: bool) -> str:
    return f"{years:.0f}y" if reached else f">{years:.0f}y (not reached)"


if __name__ == "__main__":
    run()
