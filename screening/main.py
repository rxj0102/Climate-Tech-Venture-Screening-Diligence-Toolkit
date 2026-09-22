"""Standalone entry point for Module 1: deal screening & scoring.

Run with ``python -m screening.main``. Prints the full ranked pipeline and
the shortlist so results can be pasted elsewhere for review.
"""

from __future__ import annotations

from screening.data_generator import generate_pipeline
from screening.scoring import rank_pipeline, shortlist


def run(n_deals: int = 50, seed: int = 42, shortlist_size: int = 15) -> None:
    deals = generate_pipeline(n_deals=n_deals, seed=seed)
    ranked = rank_pipeline(deals)
    top = shortlist(ranked, top_n=shortlist_size)

    print("=" * 88)
    print(f"MODULE 1 -- DEAL SCREENING & SCORING (n={len(deals)} deals, seed={seed})")
    print("=" * 88)
    print(f"\nFull ranked pipeline ({len(ranked)} deals):\n")
    for rank, result in enumerate(ranked, start=1):
        d = result.deal
        flag = f" [FLAG: {d.disqualifying_flag}]" if d.disqualifying_flag else ""
        print(
            f"{rank:>3}. {result.composite_score:>5.1f}  {d.company_name:<24} "
            f"{d.sector:<28} {d.stage:<9}{flag}"
        )

    print(f"\nShortlist (top {len(top)}):\n")
    for rank, result in enumerate(top, start=1):
        d = result.deal
        print(f"--- #{rank}: {d.company_name} ({d.deal_id}) ---")
        print(f"    Sector: {d.sector} | Stage: {d.stage} | TRL: {d.trl}")
        print(f"    Ask: ${d.ask_amount_usd:,.0f} | Pre-money: ${d.pre_money_valuation_usd:,.0f}")
        print(f"    Composite score: {result.composite_score}/100")
        print(f"    Factor contributions: {result.factor_contributions}")
        print(f"    Rationale: {result.rationale}\n")


if __name__ == "__main__":
    run()
