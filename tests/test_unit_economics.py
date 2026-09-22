from __future__ import annotations

import math

import numpy as np
import pytest

from unit_economics.learning_curve import (
    CostCurveParams,
    cumulative_volume_at_year,
    gross_margin,
    learning_exponent,
    profitability_timeline,
    unit_cost_from_v1,
    unit_cost_relative,
    years_to_margin_threshold,
)
from unit_economics.monte_carlo import MonteCarloConfig, TriangularRange, run_monte_carlo
from unit_economics.sensitivity import ParameterRange, tornado_sensitivity


def test_learning_exponent_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        learning_exponent(1.0)
    with pytest.raises(ValueError):
        learning_exponent(0.0)
    with pytest.raises(ValueError):
        learning_exponent(-0.5)


def test_unit_cost_from_v1_matches_closed_form_doubling() -> None:
    """The defining property of Wright's Law: cost(2V) == cost(V) * LR,
    for any starting volume V >= 1."""
    cost_first_unit = 10_000.0
    learning_rate = 0.8
    for v in (1.0, 5.0, 37.0, 1000.0):
        cost_v = unit_cost_from_v1(cost_first_unit, v, learning_rate)
        cost_2v = unit_cost_from_v1(cost_first_unit, 2 * v, learning_rate)
        assert math.isclose(cost_2v, cost_v * learning_rate, rel_tol=1e-9)


def test_unit_cost_from_v1_known_point() -> None:
    """At V=1, cost must equal cost_first_unit exactly."""
    assert math.isclose(unit_cost_from_v1(1234.0, 1.0, 0.85), 1234.0)


def test_unit_cost_from_v1_validates_inputs() -> None:
    with pytest.raises(ValueError):
        unit_cost_from_v1(-1.0, 10.0, 0.85)
    with pytest.raises(ValueError):
        unit_cost_from_v1(100.0, 0.5, 0.85)


def test_unit_cost_relative_matches_closed_form_doubling() -> None:
    cost_at_ref = 5000.0
    ref_volume = 100.0
    learning_rate = 0.9
    for multiplier in (1.0, 3.0, 10.0):
        v = ref_volume * multiplier
        cost_v = unit_cost_relative(cost_at_ref, ref_volume, v, learning_rate)
        cost_2v = unit_cost_relative(cost_at_ref, ref_volume, 2 * v, learning_rate)
        assert math.isclose(cost_2v, cost_v * learning_rate, rel_tol=1e-9)


def test_unit_cost_relative_equals_reference_at_reference_volume() -> None:
    cost = unit_cost_relative(5000.0, 100.0, 100.0, 0.85)
    assert math.isclose(cost, 5000.0)


def test_unit_cost_relative_validates_inputs() -> None:
    with pytest.raises(ValueError):
        unit_cost_relative(0.0, 100.0, 100.0, 0.85)
    with pytest.raises(ValueError):
        unit_cost_relative(100.0, 0.5, 100.0, 0.85)
    with pytest.raises(ValueError):
        unit_cost_relative(100.0, 100.0, 0.5, 0.85)


def test_cumulative_volume_at_year_growth() -> None:
    assert cumulative_volume_at_year(100.0, 0.30, 0) == 100.0
    assert math.isclose(cumulative_volume_at_year(100.0, 0.30, 1), 130.0)
    assert math.isclose(cumulative_volume_at_year(100.0, 0.0, 10), 100.0)


def test_cumulative_volume_at_year_validates_inputs() -> None:
    with pytest.raises(ValueError):
        cumulative_volume_at_year(0.5, 0.3, 0)
    with pytest.raises(ValueError):
        cumulative_volume_at_year(100.0, -1.5, 0)
    with pytest.raises(ValueError):
        cumulative_volume_at_year(100.0, 0.3, -1)


def test_gross_margin_basic() -> None:
    assert math.isclose(gross_margin(100.0, 60.0), 0.4)
    assert gross_margin(100.0, 150.0) < 0
    with pytest.raises(ValueError):
        gross_margin(0.0, 10.0)


def _base_params() -> CostCurveParams:
    return CostCurveParams(
        starting_cost_per_unit=9000.0,
        v0=100.0,
        growth_rate=0.30,
        learning_rate=0.85,
        price_per_unit=3500.0,
    )


def test_profitability_timeline_length_and_monotonic_cost() -> None:
    timeline = profitability_timeline(_base_params(), max_years=20)
    assert len(timeline) == 21
    costs = [s.unit_cost for s in timeline]
    assert all(costs[i] > costs[i + 1] for i in range(len(costs) - 1))


def test_years_to_margin_threshold_matches_timeline() -> None:
    params = _base_params()
    breakeven = years_to_margin_threshold(params, target_margin=0.0, max_years=25)
    target = years_to_margin_threshold(params, target_margin=0.30, max_years=25)
    assert breakeven is not None
    assert target is not None
    assert target > breakeven

    timeline = profitability_timeline(params, max_years=25)
    for s in timeline:
        if s.year == breakeven:
            assert s.gross_margin >= 0.0
        elif s.year == breakeven - 1:
            assert s.gross_margin < 0.0


def test_years_to_margin_threshold_none_when_unreachable() -> None:
    params = CostCurveParams(
        starting_cost_per_unit=100_000.0,
        v0=10.0,
        growth_rate=0.01,
        learning_rate=0.99,
        price_per_unit=10.0,
    )
    assert years_to_margin_threshold(params, target_margin=0.0, max_years=5) is None


def test_monte_carlo_percentiles_are_ordered_and_sane() -> None:
    result = run_monte_carlo(MonteCarloConfig(n_paths=3000, seed=1))
    values = [result.percentiles[p] for p in (10, 25, 50, 75, 90)]
    assert values == sorted(values)
    assert 0.0 <= result.pct_never_profitable <= 1.0
    assert len(result.years_to_profit) == 3000


def test_monte_carlo_all_censored_when_never_profitable() -> None:
    config = MonteCarloConfig(
        n_paths=200,
        seed=1,
        v0=10.0,
        price_per_unit=1.0,
        max_years=3,
        starting_cost_per_unit=TriangularRange(999_999.0, 1_000_000.0, 1_000_001.0),
        learning_rate=TriangularRange(0.989, 0.99, 0.991),
        growth_rate=TriangularRange(0.009, 0.01, 0.011),
    )
    result = run_monte_carlo(config)
    assert result.pct_never_profitable == 1.0
    assert math.isnan(result.mean_years)
    assert all(math.isnan(v) for v in result.percentiles.values())


def test_monte_carlo_converges_with_more_paths() -> None:
    """Percentile estimates from a large sample should be close to those
    from an even larger sample using the same seed and parameters, as a
    convergence sanity check (not exact equality, since sample size differs)."""
    small = run_monte_carlo(MonteCarloConfig(n_paths=4000, seed=99))
    large = run_monte_carlo(MonteCarloConfig(n_paths=20000, seed=99))
    for p in (10, 25, 50, 75, 90):
        assert abs(small.percentiles[p] - large.percentiles[p]) <= 2.0


def test_monte_carlo_reproducible_with_same_seed() -> None:
    a = run_monte_carlo(MonteCarloConfig(n_paths=1000, seed=7))
    b = run_monte_carlo(MonteCarloConfig(n_paths=1000, seed=7))
    assert np.array_equal(a.years_to_profit, b.years_to_profit)


def test_triangular_range_validates_order() -> None:
    with pytest.raises(ValueError):
        TriangularRange(low=5.0, mode=1.0, high=10.0)


def test_tornado_sensitivity_ranks_by_swing_descending() -> None:
    ranges = {
        "learning_rate": ParameterRange(low=0.72, high=0.95),
        "growth_rate": ParameterRange(low=0.10, high=0.55),
        "starting_cost_per_unit": ParameterRange(low=6000.0, high=13000.0),
    }
    results = tornado_sensitivity(_base_params(), ranges, target_margin=0.0, max_years=25)
    swings = [r.swing for r in results]
    assert swings == sorted(swings, reverse=True)
    assert {r.parameter for r in results} == set(ranges)


def test_tornado_sensitivity_rejects_unknown_parameter() -> None:
    with pytest.raises(ValueError):
        tornado_sensitivity(_base_params(), {"not_a_field": ParameterRange(0.0, 1.0)})


def test_parameter_range_validates_order() -> None:
    with pytest.raises(ValueError):
        ParameterRange(low=10.0, high=1.0)
