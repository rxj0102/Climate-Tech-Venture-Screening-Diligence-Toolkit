"""Standalone entry point for Module 4: diligence tracker & IC memo generator.

Run with ``python -m diligence.main``. Screens and shortlists the
synthetic pipeline (Module 1), runs the diligence checklist and
rules-based recommendation engine against each shortlisted deal, and
prints the full IC memo for the deliberately borderline deal to
demonstrate the override behavior end to end.
"""

from __future__ import annotations

from diligence.checklist import generate_diligence_status, high_severity_unresolved
from diligence.deal_analysis import derive_fund_return_summary, derive_unit_economics_summary
from diligence.memo import generate_memo
from diligence.recommendation import recommend
from screening.data_generator import generate_pipeline
from screening.scoring import rank_pipeline, shortlist

SEED = 42
SHORTLIST_SIZE = 15


def run() -> None:
    print("=" * 88)
    print("MODULE 4 -- DILIGENCE TRACKER & IC MEMO GENERATOR")
    print("=" * 88)

    deals = generate_pipeline(n_deals=50, seed=SEED)
    ranked = rank_pipeline(deals)
    top = shortlist(ranked, top_n=SHORTLIST_SIZE)

    print(f"\nRecommendation summary for shortlist ({len(top)} deals):\n")
    borderline_result = None
    for result in top:
        items = generate_diligence_status(result.deal, seed=SEED)
        rec = recommend(result.composite_score, items)
        flag_marker = " <-- OVERRIDE" if rec.overridden else ""
        print(
            f"  {result.deal.company_name:<24} score={result.composite_score:>5.1f}  "
            f"decision={rec.decision:<24}{flag_marker}"
        )
        if result.deal.disqualifying_flag is not None:
            borderline_result = (result, items, rec)

    if borderline_result is None:
        raise RuntimeError("Expected the borderline deal to be present in the shortlist")

    result, items, rec = borderline_result
    print(
        f"\n{'-' * 88}\nFull IC memo for the borderline deal "
        f"({result.deal.company_name}):\n{'-' * 88}\n"
    )
    unit_econ = derive_unit_economics_summary(result.deal, seed=SEED)
    fund_return = derive_fund_return_summary(result.deal)
    memo = generate_memo(result, items, rec, unit_econ, fund_return)
    print(memo)

    blocking = high_severity_unresolved(items)
    print(f"Unresolved/flagged HIGH-severity items driving the override: {len(blocking)}")
    for item in blocking:
        print(f"  - {item.description} (status: {item.status})")


if __name__ == "__main__":
    run()
