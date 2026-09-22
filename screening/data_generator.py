"""Synthetic climate tech deal pipeline generator.

Every field produced here is fabricated by this script using a seeded
``numpy`` random generator. Nothing is scraped, looked up, or intended to
resemble any real company, fund, or transaction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SECTORS: tuple[str, ...] = (
    "long_duration_energy_storage",
    "industrial_decarbonization",
    "carbon_removal",
    "grid_software",
    "alt_proteins_materials",
)

# Sectors where the deployment cost of a physical unit is the primary driver
# of unit economics, i.e. where a Wright's Law-style hardware learning curve
# (Module 2) is a reasonable proxy at all.
HARDWARE_SECTORS: frozenset[str] = frozenset(
    {"long_duration_energy_storage", "industrial_decarbonization", "carbon_removal"}
)

STAGES: tuple[str, ...] = ("seed", "series_a")

_NAME_PREFIXES = (
    "Aeon",
    "Bedrock",
    "Cascade",
    "Delta",
    "Ecliptic",
    "Fenwick",
    "Gale",
    "Helio",
    "Ironclad",
    "Juniper",
    "Kestrel",
    "Lumen",
    "Meridian",
    "Northfield",
    "Onyx",
    "Pallas",
    "Quarry",
    "Ridgeline",
    "Solstice",
    "Terraform",
    "Umbra",
    "Vantage",
    "Wraith",
    "Xenon",
    "Yarrow",
    "Zephyr",
)

_NAME_SUFFIXES = (
    "Dynamics",
    "Systems",
    "Labs",
    "Energy",
    "Materials",
    "Works",
    "Technologies",
    "Sciences",
    "Industries",
    "Robotics",
)

_BORDERLINE_DEAL_ID = "deal-borderline-01"


@dataclass(frozen=True)
class Deal:
    """A single synthetic pipeline entry.

    All monetary fields are USD. ``burn_multiple`` is net burn divided by net
    new annualized revenue for the trailing period (lower is more capital
    efficient). ``disqualifying_flag`` is ``None`` unless the deal carries a
    single hard-stop diligence issue that a pure composite score would miss.
    """

    deal_id: str
    company_name: str
    sector: str
    stage: str
    ask_amount_usd: float
    pre_money_valuation_usd: float
    trl: int
    founder_team_score: float
    moat_score: float
    tam_usd: float
    burn_multiple: float
    policy_tailwind: bool
    disqualifying_flag: str | None = None

    @property
    def is_hardware_sector(self) -> bool:
        return self.sector in HARDWARE_SECTORS


def _make_company_name(rng: np.random.Generator, used: set[str]) -> str:
    while True:
        name = f"{rng.choice(_NAME_PREFIXES)} {rng.choice(_NAME_SUFFIXES)}"
        if name not in used:
            used.add(name)
            return name


def _sample_deal(rng: np.random.Generator, deal_id: str, used_names: set[str]) -> Deal:
    sector = str(rng.choice(SECTORS))
    stage = str(rng.choice(STAGES))
    trl = int(rng.integers(1, 10))

    ask_range = (1_500_000, 4_000_000) if stage == "seed" else (6_000_000, 18_000_000)
    ask = float(rng.uniform(*ask_range))
    pre_money = float(ask * rng.uniform(2.5, 5.5))
    tam = float(10 ** rng.uniform(8.5, 10.8))

    return Deal(
        deal_id=deal_id,
        company_name=_make_company_name(rng, used_names),
        sector=sector,
        stage=stage,
        ask_amount_usd=round(ask, -3),
        pre_money_valuation_usd=round(pre_money, -3),
        trl=trl,
        founder_team_score=round(float(rng.uniform(2.0, 9.5)), 1),
        moat_score=round(float(rng.uniform(1.0, 9.0)), 1),
        tam_usd=round(tam, -3),
        burn_multiple=round(float(rng.uniform(0.6, 5.0)), 2),
        policy_tailwind=bool(rng.random() < 0.45),
        disqualifying_flag=None,
    )


def _make_borderline_deal() -> Deal:
    """A deliberately constructed edge case for IC dynamics.

    Strong on every factor the composite scoring model rewards (team, moat,
    TAM, TRL, capital efficiency, policy tailwind) yet carries a single
    disqualifying diligence flag. A scoring model built purely on weighted
    factors will rank this deal near the top of the pipeline; the diligence
    engine in Module 4 is what is supposed to catch it.
    """
    return Deal(
        deal_id=_BORDERLINE_DEAL_ID,
        company_name="Meridian Dynamics",
        sector="long_duration_energy_storage",
        stage="series_a",
        ask_amount_usd=12_000_000.0,
        pre_money_valuation_usd=55_000_000.0,
        trl=7,
        founder_team_score=8.8,
        moat_score=8.2,
        tam_usd=9_500_000_000.0,
        burn_multiple=1.1,
        policy_tailwind=True,
        disqualifying_flag="undisclosed co-founder equity litigation",
    )


def generate_pipeline(n_deals: int = 50, seed: int = 42) -> list[Deal]:
    """Generate a synthetic deal pipeline of ``n_deals`` entries.

    The final entry is always the deliberately borderline deal described in
    :func:`_make_borderline_deal`, so the returned list always has length
    ``n_deals`` and always demonstrates the score-vs-risk-flag dynamic.
    """
    if n_deals < 2:
        raise ValueError("n_deals must be at least 2 to include the borderline case")

    rng = np.random.default_rng(seed)
    used_names: set[str] = {"Meridian Dynamics"}
    deals = [
        _sample_deal(rng, f"deal-{i:03d}", used_names) for i in range(1, n_deals)
    ]
    deals.append(_make_borderline_deal())
    return deals
