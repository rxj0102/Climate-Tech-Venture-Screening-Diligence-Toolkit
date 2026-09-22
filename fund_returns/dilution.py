"""Ownership dilution and fund-level MOIC across a sequence of financing rounds.

Simplifying assumption (documented, see README): each round dilutes every
existing shareholder proportionally by ``pre_money / post_money``. This
ignores option-pool refreshes, secondary sales, liquidation preference
stacking, and pay-to-play provisions -- all of which move real cap tables
and are out of scope for this toolkit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FinancingRound:
    """A single future financing round after the fund's initial check."""

    name: str
    pre_money_usd: float
    raise_amount_usd: float
    fund_participates: bool = False
    fund_follow_on_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.pre_money_usd <= 0 or self.raise_amount_usd <= 0:
            raise ValueError("pre_money_usd and raise_amount_usd must be positive")
        if self.fund_follow_on_usd < 0:
            raise ValueError("fund_follow_on_usd cannot be negative")
        if self.fund_follow_on_usd > 0 and not self.fund_participates:
            raise ValueError("fund_follow_on_usd requires fund_participates=True")

    @property
    def post_money_usd(self) -> float:
        return self.pre_money_usd + self.raise_amount_usd


@dataclass(frozen=True)
class OwnershipSnapshot:
    round_name: str
    ownership_pct: float
    cumulative_invested_usd: float


def initial_ownership_pct(
    investment_usd: float, entry_pre_money_usd: float, entry_raise_amount_usd: float
) -> float:
    """Ownership percentage the fund receives from its entry-round check."""
    if investment_usd <= 0:
        raise ValueError("investment_usd must be positive")
    post_money = entry_pre_money_usd + entry_raise_amount_usd
    if investment_usd > post_money:
        raise ValueError("investment_usd cannot exceed the entry round's post-money valuation")
    return investment_usd / post_money


def simulate_ownership(
    investment_usd: float,
    entry_pre_money_usd: float,
    entry_raise_amount_usd: float,
    rounds: list[FinancingRound],
) -> list[OwnershipSnapshot]:
    """Ownership timeline from the fund's entry check through each
    subsequent financing round, applying pro-rata dilution and any
    configured follow-on participation."""
    ownership = initial_ownership_pct(investment_usd, entry_pre_money_usd, entry_raise_amount_usd)
    invested = investment_usd
    snapshots = [OwnershipSnapshot("Entry", ownership, invested)]

    for round_ in rounds:
        dilution_factor = round_.pre_money_usd / round_.post_money_usd
        ownership *= dilution_factor
        if round_.fund_participates and round_.fund_follow_on_usd > 0:
            ownership += round_.fund_follow_on_usd / round_.post_money_usd
            invested += round_.fund_follow_on_usd
        snapshots.append(OwnershipSnapshot(round_.name, ownership, invested))

    return snapshots


def moic(proceeds_usd: float, invested_usd: float) -> float:
    """Multiple on invested capital."""
    if invested_usd <= 0:
        raise ValueError("invested_usd must be positive")
    return proceeds_usd / invested_usd


@dataclass(frozen=True)
class ExitOutcome:
    scenario: str
    exit_value_usd: float
    fund_ownership_pct: float
    proceeds_usd: float
    moic: float


def run_exit_scenarios(
    final_ownership_pct: float,
    total_invested_usd: float,
    exit_scenarios: dict[str, float],
) -> list[ExitOutcome]:
    """Compute fund proceeds and MOIC for each named exit-value scenario."""
    outcomes = []
    for scenario, exit_value in exit_scenarios.items():
        proceeds = final_ownership_pct * exit_value
        outcomes.append(
            ExitOutcome(
                scenario=scenario,
                exit_value_usd=exit_value,
                fund_ownership_pct=final_ownership_pct,
                proceeds_usd=proceeds,
                moic=moic(proceeds, total_invested_usd),
            )
        )
    return outcomes
