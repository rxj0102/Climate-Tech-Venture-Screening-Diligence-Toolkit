from __future__ import annotations

import math

import pytest

from screening.data_generator import HARDWARE_SECTORS, SECTORS, STAGES, Deal, generate_pipeline
from screening.scoring import WEIGHTS, rank_pipeline, score_deal, shortlist


def test_weights_sum_to_one() -> None:
    assert math.isclose(sum(WEIGHTS.values()), 1.0, abs_tol=1e-9)


def test_generate_pipeline_length_and_reproducibility() -> None:
    deals_a = generate_pipeline(n_deals=50, seed=42)
    deals_b = generate_pipeline(n_deals=50, seed=42)
    assert len(deals_a) == 50
    assert deals_a == deals_b


def test_generate_pipeline_different_seed_differs() -> None:
    deals_a = generate_pipeline(n_deals=50, seed=42)
    deals_b = generate_pipeline(n_deals=50, seed=7)
    assert deals_a[:-1] != deals_b[:-1]


def test_generate_pipeline_rejects_too_small() -> None:
    with pytest.raises(ValueError):
        generate_pipeline(n_deals=1)


def test_generate_pipeline_covers_at_least_five_sectors() -> None:
    deals = generate_pipeline(n_deals=60, seed=42)
    assert len(SECTORS) >= 5
    assert {d.sector for d in deals} <= set(SECTORS)


def test_generate_pipeline_fields_within_expected_ranges() -> None:
    deals = generate_pipeline(n_deals=50, seed=42)
    for d in deals:
        assert d.sector in SECTORS
        assert d.stage in STAGES
        assert 1 <= d.trl <= 9
        assert 0.0 <= d.founder_team_score <= 10.0
        assert 0.0 <= d.moat_score <= 10.0
        assert d.ask_amount_usd > 0
        assert d.pre_money_valuation_usd > 0
        assert d.tam_usd > 0
        assert d.burn_multiple > 0
        assert isinstance(d.policy_tailwind, bool)


def test_borderline_deal_present_and_flagged() -> None:
    deals = generate_pipeline(n_deals=50, seed=42)
    borderline = [d for d in deals if d.disqualifying_flag is not None]
    assert len(borderline) == 1
    assert borderline[0].deal_id == "deal-borderline-01"
    assert borderline[0].founder_team_score >= 8.0
    assert borderline[0].moat_score >= 8.0


def test_is_hardware_sector_property() -> None:
    hw = Deal(
        deal_id="x",
        company_name="X",
        sector="carbon_removal",
        stage="seed",
        ask_amount_usd=1.0,
        pre_money_valuation_usd=1.0,
        trl=5,
        founder_team_score=5.0,
        moat_score=5.0,
        tam_usd=1e9,
        burn_multiple=1.0,
        policy_tailwind=False,
    )
    sw = Deal(
        deal_id="y",
        company_name="Y",
        sector="grid_software",
        stage="seed",
        ask_amount_usd=1.0,
        pre_money_valuation_usd=1.0,
        trl=5,
        founder_team_score=5.0,
        moat_score=5.0,
        tam_usd=1e9,
        burn_multiple=1.0,
        policy_tailwind=False,
    )
    assert hw.sector in HARDWARE_SECTORS
    assert hw.is_hardware_sector is True
    assert sw.is_hardware_sector is False


def _make_deal(**overrides: object) -> Deal:
    defaults: dict[str, object] = dict(
        deal_id="test-deal",
        company_name="Test Co",
        sector="grid_software",
        stage="seed",
        ask_amount_usd=2_000_000.0,
        pre_money_valuation_usd=8_000_000.0,
        trl=5,
        founder_team_score=5.0,
        moat_score=5.0,
        tam_usd=1e9,
        burn_multiple=2.0,
        policy_tailwind=False,
        disqualifying_flag=None,
    )
    defaults.update(overrides)
    return Deal(**defaults)  # type: ignore[arg-type]


def test_score_deal_bounds() -> None:
    weak = _make_deal(founder_team_score=0.0, moat_score=0.0, trl=1, tam_usd=1e8, burn_multiple=5.0)
    strong = _make_deal(
        founder_team_score=10.0,
        moat_score=10.0,
        trl=9,
        tam_usd=1e11,
        burn_multiple=0.5,
        policy_tailwind=True,
    )
    weak_result = score_deal(weak)
    strong_result = score_deal(strong)
    assert 0.0 <= weak_result.composite_score <= 100.0
    assert 0.0 <= strong_result.composite_score <= 100.0
    assert strong_result.composite_score > weak_result.composite_score


def test_score_deal_monotonic_in_team_score() -> None:
    low_team = score_deal(_make_deal(founder_team_score=2.0))
    high_team = score_deal(_make_deal(founder_team_score=9.0))
    assert high_team.composite_score > low_team.composite_score


def test_score_deal_capital_efficiency_is_inverted() -> None:
    """A lower burn multiple (more efficient) must score higher, not lower."""
    efficient = score_deal(_make_deal(burn_multiple=0.6))
    inefficient = score_deal(_make_deal(burn_multiple=4.9))
    assert efficient.composite_score > inefficient.composite_score


def test_score_deal_contributions_sum_to_composite() -> None:
    result = score_deal(_make_deal())
    assert math.isclose(
        sum(result.factor_contributions.values()), result.composite_score, abs_tol=0.05
    )


def test_rationale_flags_disqualifying_issue() -> None:
    flagged = _make_deal(disqualifying_flag="undisclosed litigation")
    result = score_deal(flagged)
    assert "WARNING" in result.rationale
    assert "undisclosed litigation" in result.rationale


def test_rationale_omits_warning_when_no_flag() -> None:
    clean = _make_deal()
    result = score_deal(clean)
    assert "WARNING" not in result.rationale


def test_rank_pipeline_sorted_descending() -> None:
    deals = generate_pipeline(n_deals=40, seed=1)
    ranked = rank_pipeline(deals)
    scores = [r.composite_score for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert len(ranked) == len(deals)


def test_shortlist_respects_top_n() -> None:
    deals = generate_pipeline(n_deals=40, seed=1)
    ranked = rank_pipeline(deals)
    top = shortlist(ranked, top_n=10)
    assert len(top) == 10
    assert top == ranked[:10]


def test_borderline_deal_ranks_highly_despite_flag() -> None:
    """The whole point of the borderline deal: composite scoring alone ranks
    it near the top even though it carries a disqualifying flag. Module 4's
    diligence engine is responsible for catching this, not the scorer."""
    deals = generate_pipeline(n_deals=50, seed=42)
    ranked = rank_pipeline(deals)
    borderline_rank = next(
        i for i, r in enumerate(ranked, start=1) if r.deal.disqualifying_flag is not None
    )
    assert borderline_rank <= 5
