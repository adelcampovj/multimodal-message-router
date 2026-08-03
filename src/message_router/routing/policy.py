"""Ordered, first-match routing policy.

Exactly one action is selected, by the first rule in a fixed list whose
predicate holds. There is no weighted sum, no point total, no probability,
and no aggregate score anywhere: a "positive" feature cannot accumulate
against a risk, because nothing here accumulates at all.

Precedence, highest first:

1. ``route.safety.hard_veto`` -- a safety obligation, short-circuited
2. ``route.risk.*`` -- explicit suspicious combinations
3. ``route.urgency.safe_interrupt`` -- safe, supported, genuinely urgent
4. ``route.value.*`` -- explicit low-value or unwanted suppression
5. ``route.utility.safe_defer`` -- safe and useful, but not urgent
6. ``route.analysis.incomplete_defer`` -- ambiguous, conflicted, or degraded
7. ``route.default.defer`` -- nothing matched

Rule 1 returns before any later predicate is even called, which is what
makes the veto a genuine obligation rather than a strong preference: no
amount of trust, affinity, usefulness, or urgency can be consulted, because
none of it is reached.

Routing reads its inputs and never writes to them. Severity, classification
and features pass through untouched, and this module produces no rationale,
confidence, or evidence ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..analysis.classification import ClassificationResult, MessageClass
from ..analysis.features import (
    Completeness,
    DecisionFeatures,
    FeatureFamily,
    FeatureLevel,
    ModalityConsistency,
    SupportSource,
)
from ..domain.enums import RoutingAction
from ..domain.errors import ValidationError, ValidationIssue
from ..safety.models import FindingCategory, SafetyAssessment
from .models import ContributorEffect, RoutingContributor, RoutingDiagnostic, RoutingResult

__all__ = ["RoutingPolicy", "route", "POLICY_ORDER"]

#: Explicit qualitative ordering. Used only for local comparisons inside a
#: single rule -- never summed across families, and never consulted before
#: the hard veto.
_LEVEL_ORDER = {
    FeatureLevel.ABSENT: 0,
    FeatureLevel.UNKNOWN: 0,
    FeatureLevel.INCOMPLETE: 0,
    FeatureLevel.CONFLICTED: 0,
    FeatureLevel.LOW: 1,
    FeatureLevel.MEDIUM: 2,
    FeatureLevel.HIGH: 3,
}


def _at_least(level: FeatureLevel, floor: FeatureLevel) -> bool:
    """True when ``level`` is at or above ``floor`` in the explicit ordering.

    ``UNKNOWN``, ``CONFLICTED`` and ``INCOMPLETE`` deliberately rank with
    ``ABSENT``: none of them is positive evidence, so none of them can
    satisfy a floor that asks for support.
    """
    return _LEVEL_ORDER[level] >= _LEVEL_ORDER[floor]


def _is_weak_or_missing(level: FeatureLevel) -> bool:
    """True when a level offers no positive support."""
    return _LEVEL_ORDER[level] <= 1


def _is_established_low(level: FeatureLevel) -> bool:
    """True only when a level was *established* as low or absent.

    ``UNKNOWN`` deliberately fails this: "we could not determine any value"
    is not the same finding as "we determined there is little value", and a
    rule that suppresses on the former would suppress whenever context
    happened to be missing.
    """
    return level in (FeatureLevel.ABSENT, FeatureLevel.LOW)


#: Completeness and consistency states that make a decision materially
#: under-supported, satisfying Rule 6.
_DEGRADED_COMPLETENESS = frozenset(
    {
        Completeness.INCOMPLETE_UNAVAILABLE_BACKEND,
        Completeness.INCOMPLETE_UNSUPPORTED_MEDIA,
        Completeness.INCOMPLETE_FAILED_ANALYSIS,
        Completeness.CONFLICTED,
        Completeness.INSUFFICIENT_CONTENT,
    }
)
_DEGRADED_CONSISTENCY = frozenset(
    {
        ModalityConsistency.CONFLICTING,
        ModalityConsistency.INSUFFICIENT,
        ModalityConsistency.UNAVAILABLE,
        ModalityConsistency.UNSUPPORTED,
        ModalityConsistency.FAILED,
    }
)

_PAYMENT_CONCERNS = frozenset(
    {
        FindingCategory.COERCIVE_PAYMENT,
        FindingCategory.CREDENTIAL_SOLICITATION,
        FindingCategory.ADVANCE_FEE,
        FindingCategory.FINANCIAL_REQUEST,
    }
)
_IDENTITY_CONCERNS = frozenset({FindingCategory.IMPERSONATION, FindingCategory.DECEPTIVE_LINK})


@dataclass(frozen=True)
class _Inputs:
    """The three approved inputs, read-only."""

    assessment: SafetyAssessment
    classification: ClassificationResult
    features: DecisionFeatures

    def level(self, family: FeatureFamily) -> FeatureLevel:
        return self.features.level_of(family)


def _contributor(
    inputs: _Inputs,
    family: FeatureFamily,
    effect: ContributorEffect,
    *rule_ids: str,
    support: SupportSource | None = None,
) -> RoutingContributor:
    feature = inputs.features.get(family)
    return RoutingContributor(
        family=family,
        level=feature.level if feature else FeatureLevel.UNKNOWN,
        effect=effect,
        rule_ids=tuple(rule_ids),
        support=support,
        ambiguous=bool(feature and any(c.ambiguous for c in feature.contributors)),
        conflicted=inputs.features.modality_consistency is ModalityConsistency.CONFLICTING,
        incomplete=inputs.assessment.analysis_incomplete,
    )


# ---------------------------------------------------------------------------
# Rules. Each returns a RoutingResult or None.
# ---------------------------------------------------------------------------


def _rule_hard_veto(inputs: _Inputs) -> RoutingResult | None:
    """Rule 1. A safety obligation, consulting nothing else."""
    if not inputs.assessment.hard_veto_required:
        return None
    return RoutingResult(
        action=RoutingAction.SUPPRESS,
        policy_id="route.safety.hard_veto",
        contributors=(
            _contributor(inputs, FeatureFamily.SAFETY_RISK, ContributorEffect.SUPPORTED,
                         "SAFETY_HARD_VETO", support=SupportSource.SAFETY_FINDINGS),
        ),
        hard_veto_applied=True,
        analysis_incomplete=inputs.assessment.analysis_incomplete,
        diagnostics=(RoutingDiagnostic.HARD_VETO_SHORT_CIRCUIT, RoutingDiagnostic.SEVERITY_UNCHANGED),
    )


def _rule_pressure_suppress(inputs: _Inputs) -> RoutingResult | None:
    """Rule 2A. Suspicious risk *and* high manufactured pressure."""
    if not _at_least(inputs.level(FeatureFamily.SAFETY_RISK), FeatureLevel.MEDIUM):
        return None
    if inputs.level(FeatureFamily.MALICIOUS_PRESSURE) is not FeatureLevel.HIGH:
        return None
    return _suppress(
        inputs,
        "route.risk.pressure_suppress",
        (FeatureFamily.SAFETY_RISK, "RISK_SUSPICIOUS"),
        (FeatureFamily.MALICIOUS_PRESSURE, "PRESSURE_HIGH"),
    )


def _rule_payment_identity_suppress(inputs: _Inputs) -> RoutingResult | None:
    """Rule 2B. A payment or credential concern corroborated by identity or link deception."""
    if not _at_least(inputs.level(FeatureFamily.SAFETY_RISK), FeatureLevel.MEDIUM):
        return None
    categories = inputs.assessment.categories
    if not (categories & _PAYMENT_CONCERNS) or not (categories & _IDENTITY_CONCERNS):
        return None
    return _suppress(
        inputs,
        "route.risk.payment_identity_suppress",
        (FeatureFamily.SAFETY_RISK, "RISK_PAYMENT_CONCERN"),
        (FeatureFamily.MALICIOUS_PRESSURE, "RISK_IDENTITY_OR_LINK"),
    )


def _rule_propagation_low_value_suppress(inputs: _Inputs) -> RoutingResult | None:
    """Rule 2C. Heavily circulated content with nothing in it for this reader."""
    if inputs.level(FeatureFamily.REPETITION) is not FeatureLevel.HIGH:
        return None
    if not _is_weak_or_missing(inputs.level(FeatureFamily.RECIPIENT_RELEVANCE)):
        return None
    if not _is_weak_or_missing(inputs.level(FeatureFamily.PRACTICAL_UTILITY)):
        return None
    if not _at_least(inputs.level(FeatureFamily.SAFETY_RISK), FeatureLevel.MEDIUM):
        return None
    return _suppress(
        inputs,
        "route.risk.propagation_low_value_suppress",
        (FeatureFamily.REPETITION, "REPEAT_HIGH"),
        (FeatureFamily.RECIPIENT_RELEVANCE, "RELEVANCE_LOW"),
        (FeatureFamily.PRACTICAL_UTILITY, "UTILITY_LOW"),
    )


def _rule_safe_interrupt(inputs: _Inputs) -> RoutingResult | None:
    """Rule 3. Genuinely urgent, genuinely relevant, and not under pressure."""
    if inputs.level(FeatureFamily.LEGITIMATE_URGENCY) is not FeatureLevel.HIGH:
        return None
    if inputs.level(FeatureFamily.MALICIOUS_PRESSURE) is FeatureLevel.HIGH:
        return None
    if not _at_least(inputs.level(FeatureFamily.RECIPIENT_RELEVANCE), FeatureLevel.MEDIUM):
        return None
    if inputs.features.modality_consistency is ModalityConsistency.CONFLICTING:
        return None
    if inputs.features.completeness in _DEGRADED_COMPLETENESS:
        return None

    contributors = [
        _contributor(inputs, FeatureFamily.LEGITIMATE_URGENCY, ContributorEffect.SUPPORTED, "URGENCY_HIGH"),
        _contributor(inputs, FeatureFamily.RECIPIENT_RELEVANCE, ContributorEffect.SUPPORTED, "RELEVANCE_SUPPORTED"),
        _contributor(inputs, FeatureFamily.MALICIOUS_PRESSURE, ContributorEffect.PREVENTED, "PRESSURE_NOT_HIGH"),
    ]
    # Quiet hours and a silenced thread are recorded, and deliberately do
    # not prevent a genuinely urgent message from interrupting.
    for family in (FeatureFamily.QUIET_HOURS_IMPACT, FeatureFamily.SILENCED_CONVERSATION,
                   FeatureFamily.INTERRUPTION_COST):
        contributors.append(_contributor(inputs, family, ContributorEffect.CONTEXTUAL, "COST_NOTED"))

    return RoutingResult(
        action=RoutingAction.INTERRUPT,
        policy_id="route.urgency.safe_interrupt",
        contributors=tuple(sorted(contributors, key=lambda c: c.sort_key())),
        hard_veto_applied=False,
        analysis_incomplete=inputs.assessment.analysis_incomplete,
        diagnostics=(RoutingDiagnostic.URGENCY_ELIGIBLE,),
    )


def _rule_irrelevant_marketing_suppress(inputs: _Inputs) -> RoutingResult | None:
    """Rule 4A. A promotion with neither relevance nor usefulness."""
    if inputs.classification.selected is not MessageClass.MARKETING:
        return None
    if not _is_weak_or_missing(inputs.level(FeatureFamily.RECIPIENT_RELEVANCE)):
        return None
    if not _is_weak_or_missing(inputs.level(FeatureFamily.PRACTICAL_UTILITY)):
        return None
    return _suppress(
        inputs,
        "route.value.irrelevant_marketing_suppress",
        (FeatureFamily.RECIPIENT_RELEVANCE, "RELEVANCE_LOW"),
        (FeatureFamily.PRACTICAL_UTILITY, "UTILITY_LOW"),
        diagnostic=RoutingDiagnostic.LOW_VALUE_SUPPRESSION,
    )


def _rule_repetitive_low_value_suppress(inputs: _Inputs) -> RoutingResult | None:
    """Rule 4B. Circulated content with no usefulness, relevance, or urgency.

    Requires low value to have been *established*, not merely unknown --
    otherwise forwarding metadata alone would suppress any message whose
    context happened to be unavailable, which is exactly the "forwarding
    alone" case this rule must not fire on.
    """
    if inputs.level(FeatureFamily.REPETITION) is not FeatureLevel.HIGH:
        return None
    if not (
        _is_established_low(inputs.level(FeatureFamily.PRACTICAL_UTILITY))
        or _is_established_low(inputs.level(FeatureFamily.RECIPIENT_RELEVANCE))
    ):
        return None
    if not _is_weak_or_missing(inputs.level(FeatureFamily.PRACTICAL_UTILITY)):
        return None
    if not _is_weak_or_missing(inputs.level(FeatureFamily.RECIPIENT_RELEVANCE)):
        return None
    if not _is_weak_or_missing(inputs.level(FeatureFamily.LEGITIMATE_URGENCY)):
        return None
    return _suppress(
        inputs,
        "route.value.repetitive_low_value_suppress",
        (FeatureFamily.REPETITION, "REPEAT_HIGH"),
        (FeatureFamily.PRACTICAL_UTILITY, "UTILITY_LOW"),
        diagnostic=RoutingDiagnostic.LOW_VALUE_SUPPRESSION,
    )


def _rule_muted_low_value_suppress(inputs: _Inputs) -> RoutingResult | None:
    """Rule 4C. A silenced thread carrying nothing this reader needs."""
    if inputs.level(FeatureFamily.SILENCED_CONVERSATION) is not FeatureLevel.HIGH:
        return None
    if not _is_weak_or_missing(inputs.level(FeatureFamily.RECIPIENT_RELEVANCE)):
        return None
    if not _is_weak_or_missing(inputs.level(FeatureFamily.PRACTICAL_UTILITY)):
        return None
    if not _is_weak_or_missing(inputs.level(FeatureFamily.LEGITIMATE_URGENCY)):
        return None
    return _suppress(
        inputs,
        "route.value.muted_low_value_suppress",
        (FeatureFamily.SILENCED_CONVERSATION, "SILENCED"),
        (FeatureFamily.RECIPIENT_RELEVANCE, "RELEVANCE_LOW"),
        diagnostic=RoutingDiagnostic.LOW_VALUE_SUPPRESSION,
    )


def _rule_safe_defer(inputs: _Inputs) -> RoutingResult | None:
    """Rule 5. Safe content worth keeping, but not worth interrupting for."""
    useful = _at_least(inputs.level(FeatureFamily.PRACTICAL_UTILITY), FeatureLevel.MEDIUM)
    relevant = _at_least(inputs.level(FeatureFamily.RECIPIENT_RELEVANCE), FeatureLevel.MEDIUM)
    if not (useful or relevant):
        return None
    contributors = [
        _contributor(inputs, FeatureFamily.PRACTICAL_UTILITY, ContributorEffect.SUPPORTED, "UTILITY_PRESENT"),
        _contributor(inputs, FeatureFamily.RECIPIENT_RELEVANCE, ContributorEffect.SUPPORTED, "RELEVANCE_PRESENT"),
        _contributor(inputs, FeatureFamily.LEGITIMATE_URGENCY, ContributorEffect.PREVENTED, "URGENCY_NOT_HIGH"),
    ]
    for family in (FeatureFamily.QUIET_HOURS_IMPACT, FeatureFamily.SILENCED_CONVERSATION):
        contributors.append(_contributor(inputs, family, ContributorEffect.CONTEXTUAL, "COST_NOTED"))
    return RoutingResult(
        action=RoutingAction.DEFER,
        policy_id="route.utility.safe_defer",
        contributors=tuple(sorted(contributors, key=lambda c: c.sort_key())),
        analysis_incomplete=inputs.assessment.analysis_incomplete,
        diagnostics=(RoutingDiagnostic.SAFE_USEFUL_CONTENT,),
    )


def _rule_incomplete_defer(inputs: _Inputs) -> RoutingResult | None:
    """Rule 6. Not enough dependable information to do anything but wait."""
    conflicted = inputs.features.modality_consistency in _DEGRADED_CONSISTENCY
    degraded = inputs.features.completeness in _DEGRADED_COMPLETENESS
    ambiguous = (
        inputs.classification.ambiguous
        or inputs.level(FeatureFamily.AMBIGUITY) is FeatureLevel.HIGH
        or inputs.assessment.analysis_incomplete
    )
    if not (conflicted or degraded or ambiguous):
        return None

    diagnostics = [RoutingDiagnostic.ANALYSIS_INCOMPLETE]
    if conflicted:
        diagnostics.append(RoutingDiagnostic.MODALITY_CONFLICT)
    if inputs.classification.ambiguous:
        diagnostics.append(RoutingDiagnostic.AMBIGUOUS_CLASSIFICATION)

    return RoutingResult(
        action=RoutingAction.DEFER,
        policy_id="route.analysis.incomplete_defer",
        contributors=(
            _contributor(inputs, FeatureFamily.ANALYSIS_COMPLETENESS, ContributorEffect.SUPPORTED,
                         "ANALYSIS_DEGRADED", support=SupportSource.SAFETY_FINDINGS),
            _contributor(inputs, FeatureFamily.MODALITY_CONSISTENCY, ContributorEffect.CONTEXTUAL,
                         "MODALITY_STATE"),
            _contributor(inputs, FeatureFamily.AMBIGUITY, ContributorEffect.CONTEXTUAL, "AMBIGUITY_STATE"),
        ),
        analysis_incomplete=True,
        diagnostics=tuple(sorted(set(diagnostics), key=lambda d: d.value)),
    )


def _rule_default(inputs: _Inputs) -> RoutingResult:
    """Rule 7. Nothing matched, so wait. Claims nothing."""
    return RoutingResult(
        action=RoutingAction.DEFER,
        policy_id="route.default.defer",
        contributors=(),
        analysis_incomplete=inputs.assessment.analysis_incomplete,
        diagnostics=(RoutingDiagnostic.NO_RULE_MATCHED,),
    )


def _suppress(
    inputs: _Inputs,
    policy_id: str,
    *supporting: tuple[FeatureFamily, str],
    diagnostic: RoutingDiagnostic = RoutingDiagnostic.SUSPICIOUS_COMBINATION,
) -> RoutingResult:
    contributors = [
        _contributor(inputs, family, ContributorEffect.SUPPORTED, rule_id)
        for family, rule_id in supporting
    ]
    return RoutingResult(
        action=RoutingAction.SUPPRESS,
        policy_id=policy_id,
        contributors=tuple(sorted(contributors, key=lambda c: c.sort_key())),
        hard_veto_applied=False,
        analysis_incomplete=inputs.assessment.analysis_incomplete,
        diagnostics=(diagnostic, RoutingDiagnostic.SEVERITY_UNCHANGED),
    )


#: The precedence list. Order here *is* the policy: it is evaluated top to
#: bottom and the first match wins. This is data, so a test can assert the
#: order directly rather than inferring it from behaviour.
POLICY_ORDER: tuple[tuple[str, Callable[[_Inputs], "RoutingResult | None"]], ...] = (
    ("route.safety.hard_veto", _rule_hard_veto),
    ("route.risk.pressure_suppress", _rule_pressure_suppress),
    ("route.risk.payment_identity_suppress", _rule_payment_identity_suppress),
    ("route.risk.propagation_low_value_suppress", _rule_propagation_low_value_suppress),
    ("route.urgency.safe_interrupt", _rule_safe_interrupt),
    ("route.value.irrelevant_marketing_suppress", _rule_irrelevant_marketing_suppress),
    ("route.value.repetitive_low_value_suppress", _rule_repetitive_low_value_suppress),
    ("route.value.muted_low_value_suppress", _rule_muted_low_value_suppress),
    ("route.utility.safe_defer", _rule_safe_defer),
    ("route.analysis.incomplete_defer", _rule_incomplete_defer),
)


class RoutingPolicy:
    """Evaluates the ordered rules and returns the first match.

    Stateless and side-effect free. Inputs are read, never written.
    """

    def route(
        self,
        assessment: SafetyAssessment,
        classification: ClassificationResult,
        features: DecisionFeatures,
    ) -> RoutingResult:
        if not isinstance(assessment, SafetyAssessment):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "assessment")
        if not isinstance(classification, ClassificationResult):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "classification")
        if not isinstance(features, DecisionFeatures):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "features")

        inputs = _Inputs(assessment, classification, features)
        for _policy_id, predicate in POLICY_ORDER:
            result = predicate(inputs)
            if result is not None:
                return result
        return _rule_default(inputs)


def route(
    assessment: SafetyAssessment,
    classification: ClassificationResult,
    features: DecisionFeatures,
) -> RoutingResult:
    """Convenience wrapper around :class:`RoutingPolicy`."""
    return RoutingPolicy().route(assessment, classification, features)
