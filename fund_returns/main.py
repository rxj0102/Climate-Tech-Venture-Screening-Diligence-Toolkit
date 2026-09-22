"""Standalone entry point for Module 3: ownership dilution & fund returns.

Run with ``python -m fund_returns.main``. Walks one synthetic investment
through Series A -> D dilution and a set of exit scenarios, then runs the
separate power-law portfolio-construction exercise.
"""

from __future__ import annotations

from fund_returns.dilution import FinancingRound, run_exit_scenarios, simulate_ownership
from fund_returns.power_law import (
    PowerLawConfig,
    required_hit_rate_for_target_moic,
    simulate_portfolio,
)

ENTRY_INVESTMENT_USD = 2_000_000.0
ENTRY_PRE_MONEY_USD = 8_000_000.0
ENTRY_RAISE_USD = 2_000_000.0

FOLLOW_ON_ROUNDS = [
    FinancingRound(
        name="Series B",
        pre_money_usd=30_000_000.0,
        raise_amount_usd=10_000_000.0,
        fund_participates=True,
        fund_follow_on_usd=1_000_000.0,
    ),
    FinancingRound(name="Series C", pre_money_usd=80_000_000.0, raise_amount_usd=20_000_000.0),
    FinancingRound(name="Series D", pre_money_usd=200_000_000.0, raise_amount_usd=40_000_000.0),
]

EXIT_SCENARIOS_USD = {
    "Write-off": 0.0,
    "Modest exit": 100_000_000.0,
    "Good exit": 500_000_000.0,
    "Home run": 2_000_000_000.0,
}


def run() -> None:
    print("=" * 88)
    print("MODULE 3 -- OWNERSHIP DILUTION & FUND-RETURN MODEL")
    print("=" * 88)

    print(
        f"\nEntry: ${ENTRY_INVESTMENT_USD:,.0f} into a "
        f"${ENTRY_PRE_MONEY_USD:,.0f} pre-money round."
    )
    snapshots = simulate_ownership(
        ENTRY_INVESTMENT_USD, ENTRY_PRE_MONEY_USD, ENTRY_RAISE_USD, FOLLOW_ON_ROUNDS
    )
    print("\nOwnership timeline:")
    for s in snapshots:
        print(
            f"  {s.round_name:<10} ownership={s.ownership_pct:>7.3%}  "
            f"cumulative invested=${s.cumulative_invested_usd:,.0f}"
        )

    final = snapshots[-1]
    print(f"\nExit scenarios at final ownership {final.ownership_pct:.3%}:")
    outcomes = run_exit_scenarios(
        final.ownership_pct, final.cumulative_invested_usd, EXIT_SCENARIOS_USD
    )
    for o in outcomes:
        print(
            f"  {o.scenario:<14} exit=${o.exit_value_usd:>15,.0f}  "
            f"proceeds=${o.proceeds_usd:>14,.0f}  MOIC={o.moic:>6.2f}x"
        )

    print("\n" + "-" * 88)
    print("Power-law portfolio construction exercise")
    print("-" * 88)
    config = PowerLawConfig(fund_size_usd=150_000_000.0, avg_check_size_usd=3_000_000.0)
    print(
        f"\nFund size: ${config.fund_size_usd:,.0f}  "
        f"Avg check: ${config.avg_check_size_usd:,.0f}"
    )
    print(f"Implied number of portfolio bets: {config.num_bets}")

    target_moic = 3.0
    result = required_hit_rate_for_target_moic(config, target_moic)
    if result is not None:
        hit_rate, median_moic = result
        n_winners = hit_rate * config.num_bets
        print(
            f"\nHit rate required for median simulated fund MOIC >= {target_moic}x: "
            f"{hit_rate:.1%} (~{n_winners:.1f} of {config.num_bets} bets), "
            f"achieving median MOIC {median_moic:.2f}x"
        )
    else:
        print(f"\nNo hit rate up to the search grid's ceiling reaches {target_moic}x median MOIC.")

    for sample_hit_rate in (0.05, 0.10, 0.15):
        sim = simulate_portfolio(config, sample_hit_rate)
        pct_str = ", ".join(f"p{p}={v:.2f}x" for p, v in sim.percentiles.items())
        print(f"\nAt hit rate {sample_hit_rate:.0%}: mean MOIC={sim.mean_moic:.2f}x  {pct_str}")


if __name__ == "__main__":
    run()
