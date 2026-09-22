"""Multi-factor composite scoring model for the synthetic deal pipeline.

The model is a simple weighted sum over six normalized factors. It is
deliberately transparent (no hidden ML model) so every score can be
explained with a rationale string, mirroring how an early-stage VC analyst
would justify a ranking to an investment committee. The weights below are a
documented judgment call, not a fitted or empirically derived model -- see
the README "Assumptions & Limitations" section.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from screening.data_generator import Deal

# Composite score weights. Must sum to 1.0.
WEIGHTS: dict[str, float] = {
    "team": 0.25,
    "moat": 0.20,
    "market": 0.15,
    "trl": 0.15,
    "capital_efficiency": 0.15,
    "policy_tailwind": 0.10,
}

_TRL_MIN, _TRL_MAX = 1, 9
_BURN_MULTIPLE_MIN, _BURN_MULTIPLE_MAX = 0.5, 5.0
_TAM_LOG_MIN, _TAM_LOG_MAX = 8.0, 11.0


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _normalize_team(deal: Deal) -> float:
    return _clip01(deal.founder_team_score / 10.0)


def _normalize_moat(deal: Deal) -> float:
    return _clip01(deal.moat_score / 10.0)


def _normalize_market(deal: Deal) -> float:
    """Log-scale TAM normalization: a $1B and $10B TAM should not differ 10x
    in score the way they differ in raw dollars."""
    log_tam = math.log10(max(deal.tam_usd, 1.0))
    return _clip01((log_tam - _TAM_LOG_MIN) / (_TAM_LOG_MAX - _TAM_LOG_MIN))


def _normalize_trl(deal: Deal) -> float:
    return _clip01((deal.trl - _TRL_MIN) / (_TRL_MAX - _TRL_MIN))


def _normalize_capital_efficiency(deal: Deal) -> float:
    """Lower burn multiple is better, so this factor is inverted."""
    span = _BURN_MULTIPLE_MAX - _BURN_MULTIPLE_MIN
    return _clip01((_BURN_MULTIPLE_MAX - deal.burn_multiple) / span)


def _normalize_policy_tailwind(deal: Deal) -> float:
    return 1.0 if deal.policy_tailwind else 0.0


_NORMALIZERS = {
    "team": _normalize_team,
    "moat": _normalize_moat,
    "market": _normalize_market,
    "trl": _normalize_trl,
    "capital_efficiency": _normalize_capital_efficiency,
    "policy_tailwind": _normalize_policy_tailwind,
}


@dataclass(frozen=True)
class ScoreResult:
    deal: Deal
    composite_score: float  # 0-100
    factor_contributions: dict[str, float]  # weighted contribution, 0-100 scale each
    rationale: str


def score_deal(deal: Deal) -> ScoreResult:
    """Compute the composite score (0-100) and a human-readable rationale."""
    contributions: dict[str, float] = {}
    for factor, weight in WEIGHTS.items():
        normalized = _NORMALIZERS[factor](deal)
        contributions[factor] = normalized * weight * 100.0

    composite = sum(contributions.values())
    rationale = _build_rationale(deal, contributions, composite)
    return ScoreResult(
        deal=deal,
        composite_score=round(composite, 1),
        factor_contributions={k: round(v, 2) for k, v in contributions.items()},
        rationale=rationale,
    )


def _build_rationale(deal: Deal, contributions: dict[str, float], composite: float) -> str:
    ranked_factors = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    top_two = ranked_factors[:2]
    top_str = " and ".join(f"{name.replace('_', ' ')} ({value:.1f} pts)" for name, value in top_two)

    parts = [
        f"Composite score {composite:.1f}/100, driven primarily by {top_str}."
    ]
    if deal.policy_tailwind:
        parts.append("Benefits from an active policy/regulatory tailwind.")
    if deal.disqualifying_flag is not None:
        parts.append(
            f"WARNING: disqualifying risk flag present ('{deal.disqualifying_flag}') -- "
            "a high composite score alone does not clear this deal for investment."
        )
    return " ".join(parts)


def rank_pipeline(deals: list[Deal]) -> list[ScoreResult]:
    """Score every deal and return results sorted by composite score, descending."""
    results = [score_deal(deal) for deal in deals]
    return sorted(results, key=lambda r: r.composite_score, reverse=True)


def shortlist(ranked_results: list[ScoreResult], top_n: int = 15) -> list[ScoreResult]:
    """Return the top ``top_n`` scored deals. Assumes ``ranked_results`` is
    already sorted (as returned by :func:`rank_pipeline`)."""
    return ranked_results[:top_n]
