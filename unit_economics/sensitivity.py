"""One-at-a-time sensitivity ("tornado") analysis for the learning-curve
path-to-profitability model.

For each assumption (learning rate, deployment growth rate, starting cost),
every other assumption is held at its base-case value while the target
assumption is swept from its low to its high scenario value. The resulting
swing in "years to profitability" ranks which assumption the outcome is
most sensitive to. This is a deterministic complement to the Monte Carlo
simulation in :mod:`unit_economics.monte_carlo`, not a replacement for it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from unit_economics.learning_curve import CostCurveParams, years_to_margin_threshold

# Sentinel used when a scenario never reaches the target margin within the
# horizon, so the swing can still be computed and ranked numerically.
_NOT_REACHED_SENTINEL_OFFSET = 1


@dataclass(frozen=True)
class ParameterRange:
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("low must be <= high")


@dataclass(frozen=True)
class TornadoResult:
    parameter: str
    years_at_low: float
    years_at_high: float
    reached_at_low: bool
    reached_at_high: bool
    swing: float


def tornado_sensitivity(
    base_params: CostCurveParams,
    parameter_ranges: dict[str, ParameterRange],
    target_margin: float = 0.0,
    max_years: int = 25,
) -> list[TornadoResult]:
    """Return one :class:`TornadoResult` per swept parameter, sorted by
    absolute swing in years-to-profitability, descending (largest driver
    of variance first)."""
    valid_fields = {"starting_cost_per_unit", "v0", "growth_rate", "learning_rate"}
    unknown = set(parameter_ranges) - valid_fields
    if unknown:
        raise ValueError(f"Unknown parameter(s) for sensitivity sweep: {sorted(unknown)}")

    sentinel = max_years + _NOT_REACHED_SENTINEL_OFFSET
    results: list[TornadoResult] = []

    for name, rng in parameter_ranges.items():
        low_params = replace(base_params, **{name: rng.low})
        high_params = replace(base_params, **{name: rng.high})

        years_low = years_to_margin_threshold(low_params, target_margin, max_years)
        years_high = years_to_margin_threshold(high_params, target_margin, max_years)

        reached_low = years_low is not None
        reached_high = years_high is not None
        numeric_low = float(years_low) if years_low is not None else float(sentinel)
        numeric_high = float(years_high) if years_high is not None else float(sentinel)

        results.append(
            TornadoResult(
                parameter=name,
                years_at_low=numeric_low,
                years_at_high=numeric_high,
                reached_at_low=reached_low,
                reached_at_high=reached_high,
                swing=abs(numeric_high - numeric_low),
            )
        )

    return sorted(results, key=lambda r: r.swing, reverse=True)
