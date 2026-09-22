"""Rules-based investment recommendation engine.

The core rule mirrors a standard IC dynamic: a strong composite score is
necessary but not sufficient. Any unresolved (or flagged) HIGH-severity
diligence item vetoes an automatic "proceed" recommendation, regardless of
how high the composite score is. The override is surfaced explicitly in
the rationale rather than silently lowering the score, so an analyst
reading the memo cannot miss it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from diligence.checklist import DiligenceItem, high_severity_unresolved

Decision = Literal["PROCEED", "PROCEED_WITH_CONDITIONS", "PASS", "DO_NOT_PROCEED"]

PROCEED_THRESHOLD = 75.0
CONDITIONAL_THRESHOLD = 55.0


@dataclass(frozen=True)
class Recommendation:
    decision: Decision
    rationale: str
    overridden: bool  # True iff score alone would have supported PROCEED


def recommend(composite_score: float, diligence_items: list[DiligenceItem]) -> Recommendation:
    blocking_items = high_severity_unresolved(diligence_items)

    if blocking_items:
        would_have_proceeded = composite_score >= PROCEED_THRESHOLD
        flag_list = "; ".join(f"'{i.description}'" for i in blocking_items)
        lead_in = (
            f"OVERRIDE: composite score {composite_score:.1f}/100 would ordinarily support a "
            "PROCEED recommendation, but "
            if would_have_proceeded
            else f"Composite score {composite_score:.1f}/100. "
        )
        rationale = (
            f"{lead_in}{len(blocking_items)} unresolved HIGH-severity diligence flag(s) "
            f"block a proceed recommendation regardless of score: {flag_list}."
        )
        return Recommendation(
            decision="DO_NOT_PROCEED", rationale=rationale, overridden=would_have_proceeded
        )

    if composite_score >= PROCEED_THRESHOLD:
        decision: Decision = "PROCEED"
    elif composite_score >= CONDITIONAL_THRESHOLD:
        decision = "PROCEED_WITH_CONDITIONS"
    else:
        decision = "PASS"

    rationale = (
        f"Composite score {composite_score:.1f}/100 with no unresolved HIGH-severity "
        f"diligence flags -> {decision}."
    )
    return Recommendation(decision=decision, rationale=rationale, overridden=False)
