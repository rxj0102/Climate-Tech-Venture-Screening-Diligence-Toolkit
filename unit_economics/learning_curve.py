"""Wright's Law experience-curve model and a path-to-profitability calculator.

Wright's Law states that unit cost declines as a power-law function of
cumulative production/deployment volume: each doubling of cumulative volume
reduces unit cost by a constant fraction, the "learning rate" (LR). If
``LR = 0.85``, cost falls 15% every time cumulative volume doubles.

Formally, with cumulative volume ``V`` (V=1 at the first unit) and
first-unit cost ``C1``:

    cost(V) = C1 * V ** b,   where b = log2(LR)

This module treats a single synthetic hardware-adjacent climate technology
(e.g. a long-duration storage or direct-air-capture system) and asks: given
a starting cost, a learning rate, and a deployment growth trajectory, in
what year does gross margin cross zero, and in what year does it reach a
target margin?
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def learning_exponent(learning_rate: float) -> float:
    """Convert a learning rate (e.g. 0.85 = 15% cost decline per doubling)
    into the power-law exponent ``b`` used by :func:`unit_cost`."""
    if not (0.0 < learning_rate < 1.0):
        raise ValueError("learning_rate must be strictly between 0 and 1")
    return math.log2(learning_rate)


def unit_cost_from_v1(
    cost_first_unit: float, cumulative_volume: float, learning_rate: float
) -> float:
    """Classical Wright's Law unit cost, referenced to the cost of the very
    first unit ever produced (``cumulative_volume == 1``).

    ``cumulative_volume`` must be >= 1. This is the pure closed-form power
    law: ``cost(2V) == cost(V) * learning_rate`` for any ``V >= 1``.
    """
    if cost_first_unit <= 0:
        raise ValueError("cost_first_unit must be positive")
    if cumulative_volume < 1:
        raise ValueError("cumulative_volume must be >= 1")
    b = learning_exponent(learning_rate)
    return float(cost_first_unit * cumulative_volume**b)


def unit_cost_relative(
    cost_at_reference_volume: float,
    reference_volume: float,
    cumulative_volume: float,
    learning_rate: float,
) -> float:
    """Wright's Law unit cost expressed relative to a known cost at a
    reference (e.g. "today's") cumulative volume, rather than at the
    literal first unit. Equivalent to :func:`unit_cost_from_v1` up to a
    constant rescaling, but far easier to calibrate: most real cost-curve
    exercises start from "we know our cost today at our current volume,"
    not from the cost of unit #1.
    """
    if cost_at_reference_volume <= 0:
        raise ValueError("cost_at_reference_volume must be positive")
    if reference_volume < 1:
        raise ValueError("reference_volume must be >= 1")
    if cumulative_volume < 1:
        raise ValueError("cumulative_volume must be >= 1")
    b = learning_exponent(learning_rate)
    return float(cost_at_reference_volume * (cumulative_volume / reference_volume) ** b)


def cumulative_volume_at_year(v0: float, growth_rate: float, year: int) -> float:
    """Deployed cumulative volume at ``year`` under constant exponential
    growth, starting from ``v0`` units at year 0 (``year`` >= 0)."""
    if v0 < 1:
        raise ValueError("v0 must be >= 1")
    if growth_rate <= -1.0:
        raise ValueError("growth_rate must be > -1.0")
    if year < 0:
        raise ValueError("year must be >= 0")
    return v0 * (1.0 + growth_rate) ** year


def gross_margin(price_per_unit: float, cost_per_unit: float) -> float:
    """Gross margin as a fraction of price. Can be negative."""
    if price_per_unit <= 0:
        raise ValueError("price_per_unit must be positive")
    return (price_per_unit - cost_per_unit) / price_per_unit


@dataclass(frozen=True)
class CostCurveParams:
    """Inputs to the path-to-profitability model.

    ``starting_cost_per_unit`` is today's unit cost, observed at a
    cumulative deployed volume of ``v0`` units (not the cost of the
    literal first unit ever built -- see :func:`unit_cost_relative`).
    """

    starting_cost_per_unit: float
    v0: float
    growth_rate: float
    learning_rate: float
    price_per_unit: float


@dataclass(frozen=True)
class YearSnapshot:
    year: int
    cumulative_volume: float
    unit_cost: float
    gross_margin: float


def profitability_timeline(params: CostCurveParams, max_years: int) -> list[YearSnapshot]:
    """Year-by-year cost, volume, and margin trajectory from year 0 through
    ``max_years`` inclusive."""
    snapshots = []
    for year in range(max_years + 1):
        volume = cumulative_volume_at_year(params.v0, params.growth_rate, year)
        cost = unit_cost_relative(
            params.starting_cost_per_unit, params.v0, volume, params.learning_rate
        )
        margin = gross_margin(params.price_per_unit, cost)
        snapshots.append(
            YearSnapshot(year=year, cumulative_volume=volume, unit_cost=cost, gross_margin=margin)
        )
    return snapshots


def years_to_margin_threshold(
    params: CostCurveParams, target_margin: float = 0.0, max_years: int = 30
) -> int | None:
    """First year in which gross margin reaches or exceeds ``target_margin``.

    Returns ``None`` if the threshold is not reached within ``max_years``.
    """
    for snapshot in profitability_timeline(params, max_years):
        if snapshot.gross_margin >= target_margin:
            return snapshot.year
    return None
