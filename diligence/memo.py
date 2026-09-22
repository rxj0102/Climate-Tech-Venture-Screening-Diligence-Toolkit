"""One-page investment committee memo generator.

Pulls together the outputs of all four modules into a single markdown memo
per shortlisted deal: composite score and drivers (Module 1), unit
economics outlook (Module 2, where applicable), ownership/return math
(Module 3), and diligence status with any red flags called out explicitly
(Module 4).
"""

from __future__ import annotations

from dataclasses import dataclass

from diligence.checklist import DiligenceItem, high_severity_unresolved, summarize_by_category
from diligence.recommendation import Recommendation
from fund_returns.dilution import ExitOutcome
from screening.scoring import ScoreResult

REQUIRED_MEMO_SECTIONS: tuple[str, ...] = (
    "Deal Overview",
    "Composite Score & Rationale",
    "Unit Economics Outlook",
    "Ownership & Return Analysis",
    "Diligence Status",
    "Recommendation",
)


@dataclass(frozen=True)
class UnitEconomicsSummary:
    applicable: bool
    breakeven_year: int | None = None
    target_margin_year: int | None = None
    mc_median_years: float | None = None
    mc_pct_never_profitable: float | None = None
    note: str = ""


@dataclass(frozen=True)
class FundReturnSummary:
    initial_ownership_pct: float
    final_ownership_pct: float
    total_invested_usd: float
    exit_outcomes: tuple[ExitOutcome, ...]


def _unit_economics_section(summary: UnitEconomicsSummary) -> str:
    if not summary.applicable:
        return f"Not applicable. {summary.note}".strip()

    lines = [
        f"- Year gross margin crosses 0%: {summary.breakeven_year}",
        f"- Year gross margin reaches 30% target: {summary.target_margin_year}",
    ]
    if summary.mc_median_years is not None:
        lines.append(f"- Monte Carlo median years to profitability: {summary.mc_median_years:.1f}")
    if summary.mc_pct_never_profitable is not None:
        lines.append(
            f"- Monte Carlo probability of never reaching profitability in-horizon: "
            f"{summary.mc_pct_never_profitable:.1%}"
        )
    return "\n".join(lines)


def _ownership_section(summary: FundReturnSummary) -> str:
    lines = [
        f"- Initial ownership at entry: {summary.initial_ownership_pct:.2%}",
        f"- Projected ownership at exit (post-dilution): {summary.final_ownership_pct:.2%}",
        f"- Total capital invested (entry + follow-ons): ${summary.total_invested_usd:,.0f}",
        "",
        "| Exit scenario | Exit value | Proceeds | MOIC |",
        "|---|---|---|---|",
    ]
    for outcome in summary.exit_outcomes:
        lines.append(
            f"| {outcome.scenario} | ${outcome.exit_value_usd:,.0f} | "
            f"${outcome.proceeds_usd:,.0f} | {outcome.moic:.2f}x |"
        )
    return "\n".join(lines)


def _diligence_section(diligence_items: list[DiligenceItem]) -> str:
    category_summary = summarize_by_category(diligence_items)
    lines = ["| Category | Resolved | Unresolved | Flagged |", "|---|---|---|---|"]
    for category, counts in sorted(category_summary.items()):
        lines.append(
            f"| {category} | {counts['resolved']} | {counts['unresolved']} | {counts['flagged']} |"
        )

    red_flags = high_severity_unresolved(diligence_items)
    if red_flags:
        lines.append("\n**Red flags (unresolved/flagged HIGH severity):**")
        for item in red_flags:
            note = f" -- {item.note}" if item.note else ""
            lines.append(f"- [{item.category}] {item.description} (status: {item.status}){note}")
    else:
        lines.append("\nNo unresolved HIGH-severity diligence items.")

    return "\n".join(lines)


def generate_memo(
    score_result: ScoreResult,
    diligence_items: list[DiligenceItem],
    recommendation: Recommendation,
    unit_economics: UnitEconomicsSummary,
    fund_return: FundReturnSummary,
) -> str:
    deal = score_result.deal
    sections = [
        f"# Investment Committee Memo: {deal.company_name}",
        "",
        "## Deal Overview",
        f"- Sector: {deal.sector.replace('_', ' ')}",
        f"- Stage: {deal.stage}",
        f"- Ask: ${deal.ask_amount_usd:,.0f} at ${deal.pre_money_valuation_usd:,.0f} pre-money",
        f"- TRL: {deal.trl}/9",
        f"- Policy/regulatory tailwind: {'Yes' if deal.policy_tailwind else 'No'}",
        "",
        "## Composite Score & Rationale",
        f"- Composite score: {score_result.composite_score}/100",
        f"- {score_result.rationale}",
        "",
        "## Unit Economics Outlook",
        _unit_economics_section(unit_economics),
        "",
        "## Ownership & Return Analysis",
        _ownership_section(fund_return),
        "",
        "## Diligence Status",
        _diligence_section(diligence_items),
        "",
        "## Recommendation",
        f"**{recommendation.decision}**"
        f"{'  (OVERRIDE APPLIED)' if recommendation.overridden else ''}",
        "",
        recommendation.rationale,
    ]
    return "\n".join(sections) + "\n"
