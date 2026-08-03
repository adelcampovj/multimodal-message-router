"""Immutable results of routing one message.

A routing result says *which action was selected and which ordered rule
selected it*. It deliberately carries no user-facing rationale, no
confidence value, no evidence identifiers or ranking, no probability, and
no aggregate score -- there is no number anywhere in this module that could
be summed, weighted, or thresholded.

Contributors record what the deciding rule looked at. They are explanatory,
not arithmetic: a contributor never carries a magnitude that could be
combined with another to outweigh anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..analysis.features import FeatureFamily, FeatureLevel, SupportSource
from ..domain.enums import RoutingAction

__all__ = [
    "ContributorEffect",
    "RoutingDiagnostic",
    "RoutingContributor",
    "RoutingResult",
]


class ContributorEffect(Enum):
    """How a contributor related to the selected action."""

    #: Helped satisfy the rule that fired.
    SUPPORTED = "supported"
    #: Blocked an earlier rule from firing.
    PREVENTED = "prevented"
    #: Present and recorded, but not part of why this rule fired.
    CONTEXTUAL = "contextual"


class RoutingDiagnostic(Enum):
    """Fixed, privacy-safe labels describing the routing outcome."""

    HARD_VETO_SHORT_CIRCUIT = "hard_veto_short_circuit"
    SUSPICIOUS_COMBINATION = "suspicious_combination"
    URGENCY_ELIGIBLE = "urgency_eligible"
    URGENCY_BLOCKED_BY_PRESSURE = "urgency_blocked_by_pressure"
    URGENCY_BLOCKED_BY_CONFLICT = "urgency_blocked_by_conflict"
    URGENCY_BLOCKED_BY_RELEVANCE = "urgency_blocked_by_relevance"
    LOW_VALUE_SUPPRESSION = "low_value_suppression"
    SAFE_USEFUL_CONTENT = "safe_useful_content"
    ANALYSIS_INCOMPLETE = "analysis_incomplete"
    MODALITY_CONFLICT = "modality_conflict"
    AMBIGUOUS_CLASSIFICATION = "ambiguous_classification"
    NO_RULE_MATCHED = "no_rule_matched"
    SEVERITY_UNCHANGED = "severity_unchanged"


@dataclass(frozen=True)
class RoutingContributor:
    """One thing the deciding rule looked at. Carries no content.

    ``level`` is a qualitative state, never a magnitude. Nothing in this
    package adds, weights, or ranks these.
    """

    family: FeatureFamily
    level: FeatureLevel
    effect: ContributorEffect
    rule_ids: tuple[str, ...] = ()
    support: SupportSource | None = None
    ambiguous: bool = False
    conflicted: bool = False
    incomplete: bool = False

    def sort_key(self) -> tuple:
        return (
            self.effect.value,
            self.family.value,
            self.level.value,
            self.rule_ids,
            self.support.value if self.support else "",
        )


@dataclass(frozen=True)
class RoutingResult:
    """The selected action and why, with nothing that could be re-scored."""

    action: RoutingAction
    policy_id: str
    contributors: tuple[RoutingContributor, ...] = ()
    hard_veto_applied: bool = False
    analysis_incomplete: bool = False
    diagnostics: tuple[RoutingDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        from ..domain.errors import ValidationError, ValidationIssue

        if not isinstance(self.action, RoutingAction):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "action")
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValidationError(ValidationIssue.EMPTY_IDENTIFIER, "policy_id")

    def serialize(self) -> tuple[tuple[str, object], ...]:
        """A deterministic, content-free view suitable for comparison.

        Returned as a sorted tuple of pairs rather than a dict so two runs
        can be compared exactly without depending on insertion order.
        """
        return (
            ("action", self.action.value),
            ("analysis_incomplete", self.analysis_incomplete),
            ("contributors", tuple(
                (c.family.value, c.level.value, c.effect.value, c.rule_ids)
                for c in self.contributors
            )),
            ("diagnostics", tuple(d.value for d in self.diagnostics)),
            ("hard_veto_applied", self.hard_veto_applied),
            ("policy_id", self.policy_id),
        )
