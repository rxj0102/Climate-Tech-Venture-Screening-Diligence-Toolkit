# Climate Tech Venture Screening & Diligence Toolkit

A Python toolkit that models the core analytical workflow of an
early-stage climate tech VC analyst: screening a deal pipeline,
evaluating unit economics under technology-cost uncertainty, modeling
ownership/fund-return dynamics, and running a structured diligence
process that feeds an investment committee (IC) memo.

## This is a synthetic-data project

**Every company, deal, financial figure, and diligence finding in this
repository is fabricated by seeded, documented code.** Nothing here is
scraped, sourced from a real fund or company, or intended to resemble any
real transaction. `screening/data_generator.py` generates the entire
pipeline from a `numpy` random generator with a fixed seed. This exists to
demonstrate the *modeling workflow* an early-stage climate tech investor
uses, not to make claims about any real market, company, or deal.

## What's here

| Module | Path | What it does |
|---|---|---|
| 1. Deal screening & scoring | `screening/` | Generates a synthetic 50-deal pipeline across 5 climate tech verticals and ranks it with a documented multi-factor composite score. |
| 2. Unit economics & cost curves | `unit_economics/` | Wright's Law learning-curve model, a deterministic path-to-profitability calculator, a 5,000-path Monte Carlo simulation, and a tornado sensitivity analysis. |
| 3. Ownership & fund returns | `fund_returns/` | Multi-round dilution/ownership modeling, exit-scenario MOIC, and a power-law portfolio-construction simulator. |
| 4. Diligence tracker & IC memo | `diligence/` | A 30-item diligence checklist, a rules-based recommendation engine with an explicit high-score-doesn't-override-a-red-flag rule, and a per-deal IC memo generator. |

`main.py` runs all four modules end-to-end against the same seeded
pipeline and prints a full IC memo for every shortlisted deal. Each module
also has its own `python -m <module>.main` entry point that runs and
prints just that module's results in isolation.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+.

## Running it

```bash
python main.py                       # full end-to-end run, all 4 modules
python -m screening.main             # Module 1 only
python -m unit_economics.main        # Module 2 only
python -m fund_returns.main          # Module 3 only
python -m diligence.main             # Module 4 only
```

## Testing, linting, type checking

```bash
pytest                                # runs the full suite with coverage
ruff check .                          # lint
mypy .                                 # strict type checking
```

`pyproject.toml` wires `pytest-cov` to report coverage on every module by
default; the test suite targets >90% coverage with no skipped or
`xfail`-marked tests.

---

## Methodology & assumptions

### Module 1: deal screening & scoring

The synthetic pipeline (`screening/data_generator.py`) spans five
verticals -- long-duration energy storage, industrial decarbonization,
carbon removal, grid software, and alternative proteins/materials -- with
per-deal fields for stage, ask/pre-money, TRL, founder/team score, moat
score, TAM, burn multiple (capital efficiency proxy), and a policy
tailwind flag.

The composite score (`screening/scoring.py`) is a **transparent, weighted
linear model**, not a fitted or machine-learned one:

| Factor | Weight | Normalization |
|---|---|---|
| Founder/team signal | 25% | linear 0-10 -> 0-1 |
| Competitive moat | 20% | linear 0-10 -> 0-1 |
| Market size (TAM) | 15% | log10-scaled (a $10B TAM shouldn't score 10x a $1B TAM) |
| Technology readiness (TRL) | 15% | linear 1-9 -> 0-1 |
| Capital efficiency (burn multiple) | 15% | inverted linear (lower burn multiple is better) |
| Policy/regulatory tailwind | 10% | binary |

These weights are a **documented judgment call**, chosen to reflect a
plausible early-stage climate tech thesis (team and moat matter most;
market size and policy tailwinds matter but shouldn't dominate). They are
not derived from any empirical study of what actually predicts climate
tech venture outcomes -- see Limitations below.

**The deliberately borderline deal:** the pipeline always includes one
hand-constructed deal that scores near the top of the pipeline (strong
team, moat, TAM, TRL, capital efficiency, and a policy tailwind) but
carries a single disqualifying diligence flag. The scoring model has no
way to know about this flag -- that's the point. It's Module 4's job to
catch it, mirroring a real IC dynamic where a strong headline score is
necessary but not sufficient.

### Module 2: unit economics & technology cost-curve model

`unit_economics/learning_curve.py` implements Wright's Law: unit cost
declines as a power-law function of cumulative deployment volume. Each
doubling of cumulative volume reduces cost by a constant fraction (the
"learning rate"). The module provides both the classical form (referenced
to the cost of the literal first unit ever built) and a
reference-volume-calibrated form (referenced to "our cost today, at our
current deployed volume"), which is what the path-to-profitability model
actually uses -- it's far easier to calibrate a scenario from a known
current cost than from an unknowable first-unit cost.

Given a starting cost, a learning rate, and an exponential deployment
growth trajectory, `years_to_margin_threshold` finds the year gross margin
crosses a target (0% for breakeven, a configurable positive target for a
healthy margin).

The Monte Carlo simulation (`unit_economics/monte_carlo.py`) runs 5,000
paths sampling learning rate, deployment growth rate, and starting cost
from triangular distributions, and reports **percentiles of years-to-
profitability, not a single point estimate** -- plus the fraction of paths
that never reach profitability within the modeled horizon.

The tornado sensitivity analysis (`unit_economics/sensitivity.py`) sweeps
one assumption at a time between its low and high scenario values (others
held at base case) to rank which assumption drives the most swing in the
outcome.

### Module 3: ownership & fund-return model

`fund_returns/dilution.py` models ownership dilution across Series B/C/D
financing rounds after an initial entry check. Each round dilutes every
existing shareholder by `pre_money / post_money`; a fund can optionally
participate pro-rata with a configured follow-on check, which offsets some
of that dilution. Fund proceeds and MOIC are computed under several
exit-value scenarios.

`fund_returns/power_law.py` implements the classic "power-law portfolio
construction" exercise, kept intentionally separate from the dilution
model above: given a fund size and average check size, it computes the
implied number of portfolio bets, then simulates fund-level MOIC under an
assumed hit rate (fraction of bets that are power-law "winners" drawn from
a heavy-tailed Lomax distribution; the rest return a modest-to-zero
multiple). `required_hit_rate_for_target_moic` finds the smallest hit rate
that would clear a target fund multiple, using a common-random-numbers
technique so the result is provably monotonic in the hit rate rather than
relying on a fragile search heuristic.

### Module 4: diligence tracker & IC memo generator

`diligence/checklist.py` defines a fixed 30-item checklist across five
risk categories (technology, market, team, regulatory/policy, financial).
Each shortlisted deal gets a synthetically generated status (resolved /
unresolved / flagged) per item, deterministic given the deal's id and a
global seed. HIGH-severity items are resolved more often than
MEDIUM/LOW-severity ones, reflecting that a real diligence process
prioritizes verifying the highest-risk items first -- without that
adjustment, nearly every deal in the pipeline would trip an override,
which wouldn't look like a realistic pipeline.

The recommendation engine (`diligence/recommendation.py`) implements one
rule above all others: **any unresolved or flagged HIGH-severity diligence
item vetoes an automatic PROCEED recommendation, regardless of composite
score.** When this fires on a deal whose score alone would have supported
PROCEED, the engine marks it as an explicit override in the rationale --
it doesn't just silently produce a lower recommendation. This is the same
"a good headline number doesn't excuse an unresolved red flag" pattern
that shows up in other diligence-tracking tools, reimplemented here for
this domain.

`diligence/memo.py` assembles a one-page markdown IC memo per shortlisted
deal by pulling: composite score and rationale (Module 1), unit-economics
outlook where the deal's sector is hardware-adjacent (Module 2),
ownership/return math scaled to the deal's own ask, pre-money, and TAM
(Module 3), and the full diligence status table with any HIGH-severity red
flags spelled out (Module 4).

---

## Known limitations & where this breaks down

Being upfront about simplifications, because none of these numbers should
be mistaken for investment advice or research:

- **The scoring weights are a judgment call, not a fitted model.** There
  is no backtest showing these weights predict anything. A different
  analyst could reasonably choose different weights and get a materially
  different ranking. Treat the composite score as "a documented way to
  combine several signals," not "the probability this deal succeeds."
- **The composite score has no way to see disqualifying risk.** That's by
  design (it's what Module 4 is for), but it also means the score itself
  should never be read as a standalone recommendation.
- **Wright's Law is a reasonable proxy for hardware, and a bad one for
  everything else.** This toolkit only applies the cost-curve model to
  sectors flagged as hardware-adjacent (`HARDWARE_SECTORS` in
  `screening/data_generator.py`). Applying an experience-curve model to a
  software company's unit economics wouldn't make sense, and the memo says
  so explicitly rather than forcing a number.
- **The learning curve assumes a single, stable learning rate and a
  smooth exponential deployment trajectory.** Real technology cost curves
  bend, plateau, jump discontinuously with process innovations, and can
  reverse (supply shocks, tariffs, input material price spikes). Real
  deployment is lumpy, not smooth exponential growth.
- **The dilution model ignores option-pool refreshes, liquidation
  preference stacking, secondary sales, pay-to-play provisions, and
  down-rounds with unusual terms.** Real cap tables are messier than
  `pre_money / post_money` pro-rata dilution. This is a simplification
  chosen for clarity, not a claim that real dilution works this simply.
- **Per-deal round sizes and exit values in the IC memos are formulaically
  scaled off the deal's own entry valuation and TAM**, not modeled
  independently. This keeps every deal's numbers internally consistent and
  reproducible, but it means two deals with the same entry valuation get
  the same round-progression multiples regardless of how their business
  actually performs -- a real fund's follow-on decisions depend on
  company performance, not a formula.
- **The power-law portfolio simulator assumes every bet is the same
  check size, bets are independent (no vintage or sector clustering), and
  "winner" is a single Bernoulli draw per bet.** It is also entirely
  separate from the dilution model in the same module -- the two are not
  reconciled, and shouldn't be read as two views of the same investment.
  Real portfolio construction depends on reserves, follow-on timing, and
  correlated outcomes within a vintage (e.g., a policy shift or a
  commodity cycle can sink several portfolio companies in the same
  sub-sector at once).
- **The diligence checklist's item statuses are drawn from fixed
  probabilities, not from any actual diligence process.** The severity
  labels and category weights are also a judgment call. The override rule
  itself (HIGH-severity unresolved flag blocks PROCEED) is a genuinely
  useful pattern from real diligence tracking, but the specific checklist
  items and probabilities here are illustrative, not a real fund's
  diligence process.
- **None of this accounts for macro conditions, interest rates, or the
  fundraising environment**, all of which materially affect real
  valuations, round sizes, and exit outcomes for climate tech companies.

## License

MIT. See `LICENSE`.
