from __future__ import annotations

from diligence.checklist import (
    CHECKLIST_TEMPLATE,
    generate_diligence_status,
    high_severity_unresolved,
    summarize_by_category,
)
from diligence.deal_analysis import derive_fund_return_summary, derive_unit_economics_summary
from diligence.memo import (
    REQUIRED_MEMO_SECTIONS,
    FundReturnSummary,
    UnitEconomicsSummary,
    generate_memo,
)
from diligence.recommendation import CONDITIONAL_THRESHOLD, PROCEED_THRESHOLD, recommend
from screening.data_generator import Deal, generate_pipeline
from screening.scoring import score_deal


def _clean_deal(**overrides: object) -> Deal:
    defaults: dict[str, object] = dict(
        deal_id="clean-deal",
        company_name="Clean Co",
        sector="grid_software",
        stage="seed",
        ask_amount_usd=2_000_000.0,
        pre_money_valuation_usd=8_000_000.0,
        trl=6,
        founder_team_score=7.0,
        moat_score=7.0,
        tam_usd=2e9,
        burn_multiple=1.5,
        policy_tailwind=True,
        disqualifying_flag=None,
    )
    defaults.update(overrides)
    return Deal(**defaults)  # type: ignore[arg-type]


def test_checklist_has_between_25_and_30_items_across_5_categories() -> None:
    assert 25 <= len(CHECKLIST_TEMPLATE) <= 30
    categories = {item.category for item in CHECKLIST_TEMPLATE}
    assert categories == {"technology", "market", "team", "regulatory", "financial"}


def test_checklist_item_ids_are_unique() -> None:
    ids = [item.item_id for item in CHECKLIST_TEMPLATE]
    assert len(ids) == len(set(ids))


def test_generate_diligence_status_covers_every_template_item() -> None:
    deal = _clean_deal()
    items = generate_diligence_status(deal, seed=1)
    assert len(items) == len(CHECKLIST_TEMPLATE)
    assert {i.item_id for i in items} == {t.item_id for t in CHECKLIST_TEMPLATE}
    for item in items:
        assert item.status in ("resolved", "unresolved", "flagged")


def test_generate_diligence_status_reproducible() -> None:
    deal = _clean_deal()
    items_a = generate_diligence_status(deal, seed=7)
    items_b = generate_diligence_status(deal, seed=7)
    assert items_a == items_b


def test_generate_diligence_status_differs_by_deal_id() -> None:
    deal_a = _clean_deal(deal_id="deal-a")
    deal_b = _clean_deal(deal_id="deal-b")
    items_a = generate_diligence_status(deal_a, seed=7)
    items_b = generate_diligence_status(deal_b, seed=7)
    assert [i.status for i in items_a] != [i.status for i in items_b]


def test_disqualifying_flag_surfaces_as_flagged_high_severity_item() -> None:
    deals = generate_pipeline(n_deals=50, seed=42)
    borderline = next(d for d in deals if d.disqualifying_flag is not None)
    items = generate_diligence_status(borderline, seed=42)
    matching = [i for i in items if i.note == borderline.disqualifying_flag]
    assert len(matching) == 1
    assert matching[0].severity == "HIGH"
    assert matching[0].status == "flagged"


def test_summarize_by_category_counts_add_up() -> None:
    deal = _clean_deal()
    items = generate_diligence_status(deal, seed=3)
    summary = summarize_by_category(items)
    total = sum(sum(counts.values()) for counts in summary.values())
    assert total == len(items)


def test_high_severity_unresolved_filters_correctly() -> None:
    deal = _clean_deal()
    items = generate_diligence_status(deal, seed=3)
    blocking = high_severity_unresolved(items)
    assert all(i.severity == "HIGH" for i in blocking)
    assert all(i.status in ("unresolved", "flagged") for i in blocking)


def test_recommend_proceed_with_no_blocking_items_and_high_score() -> None:
    items = [
        # No HIGH severity items at all.
    ]
    rec = recommend(PROCEED_THRESHOLD + 5, items)
    assert rec.decision == "PROCEED"
    assert rec.overridden is False


def test_recommend_proceed_with_conditions_mid_score() -> None:
    rec = recommend((PROCEED_THRESHOLD + CONDITIONAL_THRESHOLD) / 2, [])
    assert rec.decision == "PROCEED_WITH_CONDITIONS"


def test_recommend_pass_with_low_score() -> None:
    rec = recommend(CONDITIONAL_THRESHOLD - 5, [])
    assert rec.decision == "PASS"
    assert rec.overridden is False


def test_recommend_override_blocks_proceed_despite_high_score() -> None:
    """The core rule under test: a strong composite score cannot overcome
    an unresolved HIGH-severity diligence flag."""
    deal = _clean_deal(disqualifying_flag="undisclosed litigation")
    items = generate_diligence_status(deal, seed=42)
    high_score = 95.0
    rec = recommend(high_score, items)
    assert rec.decision == "DO_NOT_PROCEED"
    assert rec.overridden is True
    assert "OVERRIDE" in rec.rationale


def test_recommend_do_not_proceed_without_override_flag_when_score_already_low() -> None:
    from diligence.checklist import DiligenceItem

    items = [
        DiligenceItem(
            item_id="team-02",
            category="team",
            description="Founder legal & background check clean",
            severity="HIGH",
            status="flagged",
        )
    ]
    rec = recommend(CONDITIONAL_THRESHOLD - 10, items)
    assert rec.decision == "DO_NOT_PROCEED"
    assert rec.overridden is False


def test_borderline_pipeline_deal_gets_overridden_end_to_end() -> None:
    deals = generate_pipeline(n_deals=50, seed=42)
    borderline = next(d for d in deals if d.disqualifying_flag is not None)
    score_result = score_deal(borderline)
    items = generate_diligence_status(borderline, seed=42)
    rec = recommend(score_result.composite_score, items)
    assert score_result.composite_score >= PROCEED_THRESHOLD
    assert rec.decision == "DO_NOT_PROCEED"
    assert rec.overridden is True


def test_derive_unit_economics_summary_hardware_sector() -> None:
    deal = _clean_deal(sector="carbon_removal")
    summary = derive_unit_economics_summary(deal, seed=42)
    assert summary.applicable is True
    assert summary.breakeven_year is None or summary.breakeven_year >= 0
    assert summary.mc_pct_never_profitable is not None
    assert 0.0 <= summary.mc_pct_never_profitable <= 1.0
    if summary.breakeven_year is not None and summary.target_margin_year is not None:
        assert summary.target_margin_year >= summary.breakeven_year


def test_derive_unit_economics_summary_non_hardware_sector() -> None:
    deal = _clean_deal(sector="grid_software")
    summary = derive_unit_economics_summary(deal, seed=42)
    assert summary.applicable is False
    assert summary.note != ""


def test_derive_fund_return_summary_internally_consistent() -> None:
    deal = _clean_deal()
    summary = derive_fund_return_summary(deal)
    assert 0.0 < summary.final_ownership_pct <= summary.initial_ownership_pct
    assert summary.total_invested_usd >= deal.ask_amount_usd
    for outcome in summary.exit_outcomes:
        assert outcome.proceeds_usd == summary.final_ownership_pct * outcome.exit_value_usd


def test_generate_memo_contains_all_required_sections() -> None:
    deal = _clean_deal()
    score_result = score_deal(deal)
    items = generate_diligence_status(deal, seed=42)
    rec = recommend(score_result.composite_score, items)
    unit_econ = derive_unit_economics_summary(deal, seed=42)
    fund_return = derive_fund_return_summary(deal)
    memo = generate_memo(score_result, items, rec, unit_econ, fund_return)
    for section in REQUIRED_MEMO_SECTIONS:
        assert f"## {section}" in memo
    assert deal.company_name in memo
    assert rec.decision in memo


def test_generate_memo_not_applicable_unit_economics_section() -> None:
    deal = _clean_deal(sector="grid_software")
    score_result = score_deal(deal)
    items = generate_diligence_status(deal, seed=42)
    rec = recommend(score_result.composite_score, items)
    unit_econ = UnitEconomicsSummary(applicable=False, note="not applicable here")
    fund_return = derive_fund_return_summary(deal)
    memo = generate_memo(score_result, items, rec, unit_econ, fund_return)
    assert "Not applicable" in memo


def test_generate_memo_shows_override_marker_when_overridden() -> None:
    deal = _clean_deal(
        founder_team_score=9.5,
        moat_score=9.0,
        trl=9,
        tam_usd=5e10,
        burn_multiple=0.6,
        disqualifying_flag="undisclosed litigation",
    )
    score_result = score_deal(deal)
    assert score_result.composite_score >= PROCEED_THRESHOLD
    items = generate_diligence_status(deal, seed=42)
    rec = recommend(score_result.composite_score, items)
    unit_econ = derive_unit_economics_summary(deal, seed=42)
    fund_return = derive_fund_return_summary(deal)
    memo = generate_memo(score_result, items, rec, unit_econ, fund_return)
    assert "OVERRIDE APPLIED" in memo
    assert "Red flags" in memo


def test_fund_return_summary_dataclass_holds_exit_outcomes() -> None:
    summary = FundReturnSummary(
        initial_ownership_pct=0.2,
        final_ownership_pct=0.15,
        total_invested_usd=3_000_000.0,
        exit_outcomes=(),
    )
    assert summary.exit_outcomes == ()
