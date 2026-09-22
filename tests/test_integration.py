"""Integration test: runs a shortlisted deal through all four modules
end-to-end (screening -> unit economics -> fund returns -> diligence ->
memo) and verifies nothing crashes and the memo contains every required
section, plus a dedicated check of the diligence override behavior."""

from __future__ import annotations

from diligence.checklist import generate_diligence_status
from diligence.deal_analysis import derive_fund_return_summary, derive_unit_economics_summary
from diligence.memo import REQUIRED_MEMO_SECTIONS, generate_memo
from diligence.recommendation import PROCEED_THRESHOLD, recommend
from screening.data_generator import generate_pipeline
from screening.scoring import rank_pipeline, shortlist

SEED = 42


def test_full_pipeline_end_to_end_produces_complete_memos() -> None:
    deals = generate_pipeline(n_deals=50, seed=SEED)
    ranked = rank_pipeline(deals)
    top = shortlist(ranked, top_n=15)
    assert len(top) == 15

    for result in top:
        diligence_items = generate_diligence_status(result.deal, seed=SEED)
        recommendation = recommend(result.composite_score, diligence_items)
        unit_econ = derive_unit_economics_summary(result.deal, seed=SEED)
        fund_return = derive_fund_return_summary(result.deal)

        memo = generate_memo(result, diligence_items, recommendation, unit_econ, fund_return)

        assert memo, "memo generation must not return an empty string"
        for section in REQUIRED_MEMO_SECTIONS:
            assert f"## {section}" in memo, f"memo for {result.deal.deal_id} missing '{section}'"
        assert result.deal.company_name in memo
        assert recommendation.decision in memo


def test_full_pipeline_borderline_deal_is_overridden_not_proceed() -> None:
    """End-to-end version of the module-level override test: the
    deliberately borderline deal from Module 1 must score highly enough
    that a naive scorer would recommend PROCEED, yet the diligence engine
    must still surface an explicit override and refuse to auto-approve it."""
    deals = generate_pipeline(n_deals=50, seed=SEED)
    ranked = rank_pipeline(deals)
    top = shortlist(ranked, top_n=15)

    borderline = next(r for r in top if r.deal.disqualifying_flag is not None)
    assert borderline.composite_score >= PROCEED_THRESHOLD

    diligence_items = generate_diligence_status(borderline.deal, seed=SEED)
    recommendation = recommend(borderline.composite_score, diligence_items)

    assert recommendation.decision == "DO_NOT_PROCEED"
    assert recommendation.overridden is True
    assert "OVERRIDE" in recommendation.rationale

    unit_econ = derive_unit_economics_summary(borderline.deal, seed=SEED)
    fund_return = derive_fund_return_summary(borderline.deal)
    memo = generate_memo(borderline, diligence_items, recommendation, unit_econ, fund_return)
    assert "OVERRIDE APPLIED" in memo
    assert "DO_NOT_PROCEED" in memo


def test_full_pipeline_reproducible_across_runs() -> None:
    """Running the whole pipeline twice with the same seed must produce
    identical composite scores and recommendations, since every random
    draw in this toolkit is seeded."""
    def run_once() -> list[tuple[str, float, str]]:
        deals = generate_pipeline(n_deals=50, seed=SEED)
        ranked = rank_pipeline(deals)
        top = shortlist(ranked, top_n=15)
        out = []
        for result in top:
            items = generate_diligence_status(result.deal, seed=SEED)
            rec = recommend(result.composite_score, items)
            out.append((result.deal.deal_id, result.composite_score, rec.decision))
        return out

    assert run_once() == run_once()
