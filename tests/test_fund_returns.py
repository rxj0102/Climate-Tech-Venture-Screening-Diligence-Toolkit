from __future__ import annotations

import math

import numpy as np
import pytest

from fund_returns.dilution import (
    FinancingRound,
    initial_ownership_pct,
    moic,
    run_exit_scenarios,
    simulate_ownership,
)
from fund_returns.power_law import (
    PowerLawConfig,
    required_hit_rate_for_target_moic,
    simulate_portfolio,
)


def test_initial_ownership_pct_basic() -> None:
    pct = initial_ownership_pct(2_000_000.0, 8_000_000.0, 2_000_000.0)
    assert math.isclose(pct, 0.2)


def test_initial_ownership_pct_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        initial_ownership_pct(0.0, 8_000_000.0, 2_000_000.0)
    with pytest.raises(ValueError):
        initial_ownership_pct(20_000_000.0, 8_000_000.0, 2_000_000.0)


def test_financing_round_validates_inputs() -> None:
    with pytest.raises(ValueError):
        FinancingRound(name="B", pre_money_usd=0.0, raise_amount_usd=1.0)
    with pytest.raises(ValueError):
        FinancingRound(name="B", pre_money_usd=1.0, raise_amount_usd=-1.0)
    with pytest.raises(ValueError):
        FinancingRound(name="B", pre_money_usd=1.0, raise_amount_usd=1.0, fund_follow_on_usd=-1.0)
    with pytest.raises(ValueError):
        FinancingRound(
            name="B",
            pre_money_usd=1.0,
            raise_amount_usd=1.0,
            fund_participates=False,
            fund_follow_on_usd=5.0,
        )


def test_financing_round_post_money() -> None:
    r = FinancingRound(name="B", pre_money_usd=30_000_000.0, raise_amount_usd=10_000_000.0)
    assert math.isclose(r.post_money_usd, 40_000_000.0)


def test_simulate_ownership_dilution_without_participation() -> None:
    rounds = [FinancingRound(name="B", pre_money_usd=30_000_000.0, raise_amount_usd=10_000_000.0)]
    snapshots = simulate_ownership(2_000_000.0, 8_000_000.0, 2_000_000.0, rounds)
    assert len(snapshots) == 2
    assert math.isclose(snapshots[0].ownership_pct, 0.2)
    # Diluted by pre/post = 30/40 = 0.75
    assert math.isclose(snapshots[1].ownership_pct, 0.2 * 0.75)
    assert snapshots[1].cumulative_invested_usd == snapshots[0].cumulative_invested_usd


def test_simulate_ownership_with_pro_rata_participation() -> None:
    rounds = [
        FinancingRound(
            name="B",
            pre_money_usd=30_000_000.0,
            raise_amount_usd=10_000_000.0,
            fund_participates=True,
            fund_follow_on_usd=1_000_000.0,
        )
    ]
    snapshots = simulate_ownership(2_000_000.0, 8_000_000.0, 2_000_000.0, rounds)
    diluted = 0.2 * (30_000_000.0 / 40_000_000.0)
    expected = diluted + 1_000_000.0 / 40_000_000.0
    assert math.isclose(snapshots[1].ownership_pct, expected)
    assert math.isclose(snapshots[1].cumulative_invested_usd, 3_000_000.0)


def test_ownership_never_exceeds_100_percent_in_realistic_scenario() -> None:
    rounds = [
        FinancingRound(name="B", pre_money_usd=30_000_000.0, raise_amount_usd=10_000_000.0),
        FinancingRound(name="C", pre_money_usd=80_000_000.0, raise_amount_usd=20_000_000.0),
        FinancingRound(name="D", pre_money_usd=200_000_000.0, raise_amount_usd=40_000_000.0),
    ]
    snapshots = simulate_ownership(2_000_000.0, 8_000_000.0, 2_000_000.0, rounds)
    for s in snapshots:
        assert 0.0 <= s.ownership_pct <= 1.0
    # Dilution without participation must be monotonically non-increasing.
    ownerships = [s.ownership_pct for s in snapshots]
    assert all(ownerships[i] >= ownerships[i + 1] for i in range(len(ownerships) - 1))


def test_moic_basic_and_invalid() -> None:
    assert math.isclose(moic(6_000_000.0, 2_000_000.0), 3.0)
    with pytest.raises(ValueError):
        moic(1.0, 0.0)


def test_run_exit_scenarios_internally_consistent() -> None:
    outcomes = run_exit_scenarios(
        final_ownership_pct=0.10,
        total_invested_usd=3_000_000.0,
        exit_scenarios={"Fail": 0.0, "Good": 500_000_000.0},
    )
    by_name = {o.scenario: o for o in outcomes}
    assert by_name["Fail"].moic == 0.0
    good = by_name["Good"]
    assert math.isclose(good.proceeds_usd, 0.10 * 500_000_000.0)
    assert math.isclose(good.moic, good.proceeds_usd / 3_000_000.0)


def test_power_law_config_num_bets() -> None:
    config = PowerLawConfig(fund_size_usd=150_000_000.0, avg_check_size_usd=3_000_000.0)
    assert config.num_bets == 50


def test_power_law_config_validates_inputs() -> None:
    with pytest.raises(ValueError):
        PowerLawConfig(fund_size_usd=0.0, avg_check_size_usd=1.0)
    with pytest.raises(ValueError):
        PowerLawConfig(fund_size_usd=100.0, avg_check_size_usd=200.0)
    with pytest.raises(ValueError):
        PowerLawConfig(fund_size_usd=100.0, avg_check_size_usd=10.0, winner_alpha=0.0)


def test_simulate_portfolio_percentiles_ordered() -> None:
    config = PowerLawConfig(
        fund_size_usd=150_000_000.0, avg_check_size_usd=3_000_000.0, n_sims=2000
    )
    result = simulate_portfolio(config, hit_rate=0.10)
    values = [result.percentiles[p] for p in (10, 25, 50, 75, 90)]
    assert values == sorted(values)
    assert len(result.fund_moics) == config.n_sims
    assert all(m >= 0 for m in result.fund_moics)


def test_simulate_portfolio_rejects_invalid_hit_rate() -> None:
    config = PowerLawConfig(fund_size_usd=100.0, avg_check_size_usd=10.0, n_sims=100)
    with pytest.raises(ValueError):
        simulate_portfolio(config, hit_rate=1.5)


def test_simulate_portfolio_higher_hit_rate_increases_median_moic() -> None:
    config = PowerLawConfig(
        fund_size_usd=150_000_000.0, avg_check_size_usd=3_000_000.0, n_sims=3000
    )
    low = simulate_portfolio(config, hit_rate=0.05)
    high = simulate_portfolio(config, hit_rate=0.20)
    assert high.percentiles[50] > low.percentiles[50]


def test_required_hit_rate_for_target_moic_finds_reasonable_value() -> None:
    config = PowerLawConfig(
        fund_size_usd=150_000_000.0, avg_check_size_usd=3_000_000.0, n_sims=3000
    )
    result = required_hit_rate_for_target_moic(config, target_moic=3.0)
    assert result is not None
    hit_rate, median_moic = result
    assert 0.0 < hit_rate <= 0.5
    assert median_moic >= 3.0


def test_required_hit_rate_returns_none_when_unreachable() -> None:
    config = PowerLawConfig(
        fund_size_usd=150_000_000.0, avg_check_size_usd=3_000_000.0, n_sims=2000
    )
    result = required_hit_rate_for_target_moic(
        config, target_moic=1000.0, hit_rate_grid=np.array([0.01, 0.05])
    )
    assert result is None


def test_required_hit_rate_grid_search_is_monotonic() -> None:
    """The common-random-numbers construction guarantees that increasing
    hit rate never decreases the simulated median MOIC."""
    config = PowerLawConfig(
        fund_size_usd=150_000_000.0, avg_check_size_usd=3_000_000.0, n_sims=2000
    )
    from fund_returns.power_law import _simulate_fund_moics

    rng = np.random.default_rng(config.seed)
    shape = (config.n_sims, config.num_bets)
    common_uniforms = rng.random(shape)
    medians = []
    for hr in (0.02, 0.05, 0.10, 0.20, 0.30):
        moics = _simulate_fund_moics(config, hr, common_uniforms=common_uniforms)
        medians.append(float(np.median(moics)))
    assert medians == sorted(medians)
