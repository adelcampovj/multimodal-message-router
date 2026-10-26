"""Routing-relevant features, extracted without deciding anything.

Each feature family reports a qualitative state and the reasons behind it.
No family produces a score, a threshold, a confidence number, or an action.
Their job is to make the inputs of a later routing decision explicit and
inspectable.

Three separations are load-bearing:

* **Legitimate urgency is not malicious pressure.** A fire alarm and a
  scammer's countdown are both "urgent"; conflating them lets an attacker
  buy priority by shouting. Pressure derives only from safety findings;
  urgency derives only from independently supported signals, and safety
  findings can never raise it.
* **Risk is projected, never recomputed.** Severity and the hard-veto
  obligation pass through exactly as the safety layer reported them.
  Nothing here promotes a severity, and nothing -- trust, affinity,
  usefulness, engagement, quiet hours, silenced threads -- lowers one.
* **Trust comes only from validated structure.** Never from a name, a
  logo, extracted text, or a claim the content makes about itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..domain.context import (
    ConversationContext,
    RecipientPreferences,
    RecipientScope,
    SenderContext,
)
from ..domain.enums import Provenance
from ..domain.errors import ValidationError, ValidationIssue
from ..domain.message import IncomingMessage
from ..safety.models import (
    ConflictKind,
    DiagnosticCode,
    FindingCategory,
    ModalityState,
    SafetyAssessment,
    Severity,
)
from .classification import ACTIVE_DISRUPTION_RULE_ID

__all__ = [
    "FeatureFamily",
    "FeatureLevel",
    "SupportSource",
    "FeatureNote",
    "ModalityConsistency",
    "Completeness",
    "ContentAvailability",
    "RiskFamily",
    "RiskRootCause",
    "FeatureContributor",
    "Feature",
    "DecisionFeatures",
    "extract_features",
]


class FeatureFamily(Enum):
    """The routing-relevant dimensions this layer reports."""

    LEGITIMATE_URGENCY = "legitimate_urgency"
    MALICIOUS_PRESSURE = "malicious_pressure"
    RECIPIENT_RELEVANCE = "recipient_relevance"
    PRACTICAL_UTILITY = "practical_utility"
    VALIDATED_TRUST = "validated_trust"
    SAFETY_RISK = "safety_risk"
    REPETITION = "repetition"
    RECIPIENT_AFFINITY = "recipient_affinity"
    INTERRUPTION_COST = "interruption_cost"
    QUIET_HOURS_IMPACT = "quiet_hours_impact"
    SILENCED_CONVERSATION = "silenced_conversation"
    MODALITY_CONSISTENCY = "modality_consistency"
    ANALYSIS_COMPLETENESS = "analysis_completeness"
    AMBIGUITY = "ambiguity"
    CONTENT_AVAILABILITY = "content_availability"


class FeatureLevel(Enum):
    """Qualitative state of a feature.

    ``UNKNOWN`` means nothing established it, which is deliberately
    different from ``ABSENT`` (established as not present). Collapsing the
    two would turn missing information into a positive finding.
    """

    ABSENT = "absent"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"
    CONFLICTED = "conflicted"
    INCOMPLETE = "incomplete"


class SupportSource(Enum):
    """Where a contributor's support came from.

    Untrusted content and validated structure are never merged, so a reader
    can always tell whether a feature rests on something the sender wrote.
    """

    STRUCTURED_METADATA = "structured_metadata"
    RECIPIENT_PREFERENCES = "recipient_preferences"
    VALIDATED_HISTORY = "validated_history"
    SAFETY_FINDINGS = "safety_findings"
    UNTRUSTED_CONTENT = "untrusted_content"


class FeatureNote(Enum):
    """Fixed, privacy-safe labels qualifying a contributor."""

    NO_SUPPORTING_STRUCTURE = "no_supporting_structure"
    URGENCY_WITHHELD_RISK_ATTRIBUTED = "urgency_withheld_risk_attributed"
    QUIET_HOURS_UNKNOWN = "quiet_hours_unknown"
    PREFERENCES_UNAVAILABLE = "preferences_unavailable"
    NO_SECONDARY_MODALITY = "no_secondary_modality"
    MEDIA_ANALYSIS_DEGRADED = "media_analysis_degraded"
    MODALITY_CONFLICT = "modality_conflict"
    INSUFFICIENT_COMPARISON = "insufficient_comparison"
    NO_HISTORY_SUMMARY = "no_history_summary"
    SEVERE_FINDING_PRESENT = "severe_finding_present"
    HARD_VETO_OBLIGATION = "hard_veto_obligation"
    ANALYSIS_INCOMPLETE = "analysis_incomplete"
    NO_USABLE_CONTENT = "no_usable_content"
    CLAIM_NOT_INDEPENDENTLY_SUPPORTED = "claim_not_independently_supported"


class ModalityConsistency(Enum):
    CONSISTENT = "consistent"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class Completeness(Enum):
    COMPLETE_FOR_AVAILABLE_INPUTS = "complete_for_available_inputs"
    PARTIALLY_COMPLETE = "partially_complete"
    INCOMPLETE_UNAVAILABLE_BACKEND = "incomplete_unavailable_backend"
    INCOMPLETE_UNSUPPORTED_MEDIA = "incomplete_unsupported_media"
    INCOMPLETE_FAILED_ANALYSIS = "incomplete_failed_analysis"
    CONFLICTED = "conflicted"
    INSUFFICIENT_CONTENT = "insufficient_content"


class ContentAvailability(Enum):
    AUTHORED_ONLY = "authored_only"
    EXTRACTED_ONLY = "extracted_only"
    AUTHORED_AND_EXTRACTED = "authored_and_extracted"
    STRUCTURED_METADATA_ONLY = "structured_metadata_only"
    NO_USABLE_CONTENT = "no_usable_content"
    CONFLICTED_CONTENT = "conflicted_content"


class RiskFamily(Enum):
    """Independent kinds of corroborated risk observed in one message.

    Projected from safety findings by their fixed diagnostic codes, never
    recomputed from content. Two findings that express the same underlying
    indicator map to the same family and therefore count once, which is what
    stops a single pattern from looking like independent corroboration.

    Every member names something a message *does*. There is deliberately no
    member for "the same thing was seen twice": a second source repeating
    one indicator is a reason to believe the indicator, not a second
    indicator, and admitting it here would let one pattern complete a
    two-family suppression by being read twice.

    :attr:`is_substantive` marks the families that can carry a suppression
    on their own account. The rest -- manufactured pressure, a domain that
    merely looks wrong -- corroborate a substantive family and never stand
    in for one.
    """

    MALICIOUS_PRESSURE = "malicious_pressure"
    COERCIVE_PAYMENT = "coercive_payment"
    INFORMATION_REQUEST = "information_request"
    DECEPTIVE_LINK = "deceptive_link"
    IDENTITY_INCONSISTENCY = "identity_inconsistency"
    ADVANCE_FEE = "advance_fee"
    UNUSUAL_PAYMENT_CHANNEL = "unusual_payment_channel"

    @property
    def is_substantive(self) -> bool:
        return self not in _CORROBORATING_ONLY


_CORROBORATING_ONLY = frozenset(
    {RiskFamily.MALICIOUS_PRESSURE, RiskFamily.DECEPTIVE_LINK}
)

#: Fixed diagnostic-to-family map. ``ARTIFICIAL_TIME_PRESSURE`` is
#: deliberately absent: urgency wording on its own is not a risk family, and
#: including it would let a countdown corroborate itself.
_DIAGNOSTIC_RISK_FAMILIES = {
    DiagnosticCode.PAYMENT_DEMAND: RiskFamily.COERCIVE_PAYMENT,
    DiagnosticCode.OFF_CHANNEL_PAYMENT: RiskFamily.UNUSUAL_PAYMENT_CHANNEL,
    DiagnosticCode.CONSEQUENCE_THREAT: RiskFamily.MALICIOUS_PRESSURE,
    DiagnosticCode.SECRECY_PRESSURE: RiskFamily.MALICIOUS_PRESSURE,
    DiagnosticCode.ANTI_VERIFICATION_PRESSURE: RiskFamily.MALICIOUS_PRESSURE,
    DiagnosticCode.ADVANCE_FEE_OFFER: RiskFamily.ADVANCE_FEE,
    DiagnosticCode.CREDENTIAL_DISCLOSURE_REQUEST: RiskFamily.INFORMATION_REQUEST,
    DiagnosticCode.CARD_DETAIL_REQUEST: RiskFamily.INFORMATION_REQUEST,
    DiagnosticCode.BANKING_ACCESS_REQUEST: RiskFamily.INFORMATION_REQUEST,
    DiagnosticCode.IDENTITY_CLAIM_UNSUPPORTED: RiskFamily.IDENTITY_INCONSISTENCY,
    DiagnosticCode.LOOKALIKE_HOST: RiskFamily.DECEPTIVE_LINK,
    DiagnosticCode.RAW_IP_HOST: RiskFamily.DECEPTIVE_LINK,
    DiagnosticCode.EMBEDDED_CREDENTIALS_IN_URL: RiskFamily.DECEPTIVE_LINK,
    DiagnosticCode.PUNYCODE_HOST: RiskFamily.DECEPTIVE_LINK,
    DiagnosticCode.DISPLAY_TARGET_MISMATCH: RiskFamily.DECEPTIVE_LINK,
    DiagnosticCode.LINK_WITH_CREDENTIAL_REQUEST: RiskFamily.DECEPTIVE_LINK,
    DiagnosticCode.LINK_WITH_PAYMENT_PRESSURE: RiskFamily.DECEPTIVE_LINK,
}


@dataclass(frozen=True)
class RiskRootCause:
    """One underlying indicator, identified without reference to content.

    The key is the risk family, the safety rule that fired, and its fixed
    diagnostic code -- all closed vocabularies. No message text, URL,
    transcript, or secret-bearing value is stored or compared, so two
    findings are "the same cause" because the same rule fired for the same
    reason, never because their text looked alike.

    Provenance is deliberately **not** part of the key. That is the whole
    point: one indicator read twice, once from a caption and once from the
    image it was printed on, is one reason to be suspicious, not two.
    """

    family: "RiskFamily"
    rule_id: str
    diagnostic: DiagnosticCode

    def sort_key(self) -> tuple:
        return (self.family.value, self.rule_id, self.diagnostic.value)


def _risk_root_causes(assessment: SafetyAssessment) -> frozenset[RiskRootCause]:
    """Project the distinct underlying indicators. Severity is read, never changed.

    Only findings the safety layer reported as at least suspicious *and* not
    ambiguous qualify: a weak mention and a hedged, quoted, or educational
    match are exactly the material that must not accumulate into a
    suppression.

    Findings themselves are untouched -- none is merged, dropped, or
    rewritten, and every one keeps its own provenance for audit. What is
    deduplicated here is corroboration *credit*, not evidence.
    """
    return frozenset(
        RiskRootCause(
            family=_DIAGNOSTIC_RISK_FAMILIES[finding.diagnostic],
            rule_id=finding.rule_id,
            diagnostic=finding.diagnostic,
        )
        for finding in assessment.findings
        if finding.is_corroborating_evidence
        and finding.diagnostic in _DIAGNOSTIC_RISK_FAMILIES
    )


def _risk_families(root_causes: frozenset[RiskRootCause]) -> frozenset[RiskFamily]:
    """The families the distinct root causes belong to.

    Derived from the deduplicated causes, so repeating one indicator across
    modalities cannot widen this set.
    """
    return frozenset(cause.family for cause in root_causes)


def _cross_modal_corroboration(assessment: SafetyAssessment) -> bool:
    """Whether two *different* qualifying indicators came from different sources.

    Recorded for auditability only. It is deliberately not a risk family and
    routing never reads it: a second modality repeating the same thing is
    not a second reason, and this must never be what completes a two-family
    suppression.
    """
    seen: dict[RiskRootCause, set] = {}
    for finding in assessment.findings:
        if finding.severity.rank < Severity.SUSPICIOUS.rank or finding.ambiguous:
            continue
        if finding.diagnostic not in _DIAGNOSTIC_RISK_FAMILIES:
            continue
        cause = RiskRootCause(
            family=_DIAGNOSTIC_RISK_FAMILIES[finding.diagnostic],
            rule_id=finding.rule_id,
            diagnostic=finding.diagnostic,
        )
        seen.setdefault(cause, set()).add(finding.provenance)
    if len(seen) < 2:
        return False
    return len({p for provenances in seen.values() for p in provenances}) >= 2


@dataclass(frozen=True)
class FeatureContributor:
    """One reason a feature holds the value it does. Carries no content."""

    rule_id: str
    source: SupportSource
    provenance: Provenance
    note: FeatureNote | None = None
    ambiguous: bool = False
    analysis_unavailable: bool = False

    @property
    def from_untrusted_content(self) -> bool:
        return self.source is SupportSource.UNTRUSTED_CONTENT

    def sort_key(self) -> tuple:
        return (
            self.rule_id,
            self.source.value,
            _PROVENANCE_ORDER.get(self.provenance, 99),
            self.note.value if self.note else "",
        )


_PROVENANCE_ORDER = {member: index for index, member in enumerate(Provenance)}


@dataclass(frozen=True)
class Feature:
    """One family's qualitative state plus its reasons."""

    family: FeatureFamily
    level: FeatureLevel
    contributors: tuple[FeatureContributor, ...] = ()
    detail: str | None = None

    @property
    def rests_on_untrusted_content(self) -> bool:
        return any(c.from_untrusted_content for c in self.contributors)


@dataclass(frozen=True)
class DecisionFeatures:
    """Every feature family, plus the projected safety facts.

    Deliberately absent, and asserted absent by a structural test: routing
    action, route score, thresholds, confidence, rationale, and evidence
    ranking. The safety fields are a faithful projection -- they are copied
    from the assessment, never recomputed.
    """

    features: tuple[Feature, ...] = ()
    highest_severity: Severity | None = None
    hard_veto_required: bool = False
    risk_categories: frozenset[FindingCategory] = frozenset()
    risk_families: frozenset[RiskFamily] = frozenset()
    risk_root_causes: frozenset["RiskRootCause"] = frozenset()
    cross_modal_corroboration: bool = False
    modality_consistency: ModalityConsistency = ModalityConsistency.NOT_APPLICABLE
    completeness: Completeness = Completeness.COMPLETE_FOR_AVAILABLE_INPUTS
    content_availability: ContentAvailability = ContentAvailability.NO_USABLE_CONTENT
    #: True when legitimate urgency is ``high`` only because of content past
    #: the safety layer's analysis limit -- the same derivation from
    #: safety-analysed support alone does not reach ``high``. Never true for
    #: a message whose sources were all analysed in full.
    urgency_outside_safety_coverage: bool = False

    def get(self, family: FeatureFamily) -> Feature | None:
        for feature in self.features:
            if feature.family is family:
                return feature
        return None

    def level_of(self, family: FeatureFamily) -> FeatureLevel:
        feature = self.get(family)
        return feature.level if feature else FeatureLevel.UNKNOWN


# ---------------------------------------------------------------------------
# Extraction.
# ---------------------------------------------------------------------------

_PRESSURE_CATEGORIES = frozenset(
    {
        FindingCategory.MALICIOUS_URGENCY,
        FindingCategory.COERCIVE_PAYMENT,
        FindingCategory.ADVANCE_FEE,
    }
)

_UTILITY_CLASSES = ("SERVICE_UPDATE", "FINANCIAL_NOTICE", "SCHEDULED_EVENT", "DIRECT_REQUEST", "SAFETY_ALERT")

#: Classes that can *originate* legitimate urgency.
#:
#: ``DIRECT_REQUEST`` is deliberately absent. Being asked something is not
#: the same as being asked something urgently: request grammar, politeness,
#: capitalization, punctuation, and the isolated word "urgent" are all
#: things anyone can write, so treating a request as urgency-bearing hands
#: out priority for phrasing. A request becomes urgent only when an
#: independent signal -- a near-term event, an active hazard, an active
#: relevant disruption -- says so.
_URGENT_CLASSES = ("SAFETY_ALERT", "SCHEDULED_EVENT")

#: Classes and signals whose urgency stands alone rather than needing a
#: second supporting contributor.
_SELF_SUFFICIENT_URGENCY = ("SAFETY_ALERT",)


def extract_features(
    message: IncomingMessage,
    assessment: SafetyAssessment,
    classification=None,
    preferences: RecipientPreferences | None = None,
    interaction_summary: "InteractionSummary | None" = None,
    local_minute_of_day: int | None = None,
    forwarding_count: int | None = None,
    directly_addressed: bool = False,
    near_term_event: bool = False,
) -> DecisionFeatures:
    """Extract routing-relevant features. Chooses no route.

    ``local_minute_of_day`` is supplied explicitly rather than read from a
    clock. Quiet hours are a local-time concept and the domain records no
    recipient timezone, so guessing one would be inventing information; if
    it is not supplied, quiet-hours impact is ``UNKNOWN``.
    """
    if not isinstance(message, IncomingMessage):
        raise ValidationError(ValidationIssue.WRONG_TYPE, "message")
    if not isinstance(assessment, SafetyAssessment):
        raise ValidationError(ValidationIssue.WRONG_TYPE, "assessment")
    if forwarding_count is not None and (
        isinstance(forwarding_count, bool) or not isinstance(forwarding_count, int)
    ):
        raise ValidationError(ValidationIssue.WRONG_TYPE, "forwarding_count")
    if local_minute_of_day is not None and (
        isinstance(local_minute_of_day, bool)
        or not isinstance(local_minute_of_day, int)
        or not 0 <= local_minute_of_day < 24 * 60
    ):
        raise ValidationError(ValidationIssue.OUT_OF_RANGE, "local_minute_of_day")

    features: list[Feature] = []
    selected = getattr(classification, "selected", None)
    selected_name = selected.name if selected is not None else ""
    classification_ambiguous = bool(getattr(classification, "ambiguous", False))

    # -- safety projection (copied, never recomputed) ---------------------
    severity = assessment.highest_severity
    veto = assessment.hard_veto_required
    risk_notes: list[FeatureContributor] = []
    if severity is Severity.SEVERE:
        risk_notes.append(
            FeatureContributor("RISK_SEVERE", SupportSource.SAFETY_FINDINGS,
                               Provenance.DERIVED_SIGNAL, FeatureNote.SEVERE_FINDING_PRESENT)
        )
    if veto:
        risk_notes.append(
            FeatureContributor("RISK_VETO", SupportSource.SAFETY_FINDINGS,
                               Provenance.DERIVED_SIGNAL, FeatureNote.HARD_VETO_OBLIGATION)
        )
    risk_level = {
        Severity.SEVERE: FeatureLevel.HIGH,
        Severity.SUSPICIOUS: FeatureLevel.MEDIUM,
        Severity.WEAK: FeatureLevel.LOW,
        None: FeatureLevel.ABSENT,
    }[severity]
    features.append(Feature(FeatureFamily.SAFETY_RISK, risk_level, tuple(risk_notes)))

    # -- malicious pressure (only ever from safety findings) --------------
    pressure_findings = [f for f in assessment.findings if f.category in _PRESSURE_CATEGORIES]
    pressure_level = FeatureLevel.ABSENT
    if pressure_findings:
        strongest = max(f.severity.rank for f in pressure_findings)
        pressure_level = {0: FeatureLevel.LOW, 1: FeatureLevel.MEDIUM, 2: FeatureLevel.HIGH}[strongest]
    features.append(
        Feature(
            FeatureFamily.MALICIOUS_PRESSURE,
            pressure_level,
            tuple(
                FeatureContributor(f"PRESSURE_{f.category.name}", SupportSource.SAFETY_FINDINGS,
                                   f.provenance, ambiguous=f.ambiguous)
                for f in sorted(pressure_findings, key=lambda f: f.sort_key())
            ),
        )
    )

    # -- legitimate urgency (never raised by a safety finding) ------------
    urgency_level, urgency_contributors = _legitimate_urgency(
        selected_name, _active_disruption_contributor(classification),
        near_term_event, directly_addressed, bool(pressure_findings),
    )
    features.append(
        Feature(FeatureFamily.LEGITIMATE_URGENCY, urgency_level,
                tuple(sorted(urgency_contributors, key=lambda c: c.sort_key())))
    )
    # The same derivation from safety-analysed support alone. The feature
    # above is unchanged; this only records whether its HIGH level rests on
    # content the safety layer never read. Structured context is kept as is:
    # it is not content, so coverage does not apply to it.
    covered_class = getattr(classification, "selected_within_safety_coverage", selected)
    covered_level, _ = _legitimate_urgency(
        covered_class.name if covered_class is not None else "",
        _active_disruption_contributor(classification, within_safety_coverage=True),
        near_term_event, directly_addressed, bool(pressure_findings),
    )
    urgency_outside_safety_coverage = (
        urgency_level is FeatureLevel.HIGH and covered_level is not FeatureLevel.HIGH
    )

    # -- validated trust (structured context only) ------------------------
    trust_contributors: list[FeatureContributor] = []
    sender: SenderContext | None = message.sender
    if sender is not None and sender.has_asserted_trust:
        for signal in sender.trust_signals:
            trust_contributors.append(
                FeatureContributor(f"TRUST_{signal.category.name}", SupportSource.STRUCTURED_METADATA,
                                   signal.asserted_by)
            )
    if interaction_summary is not None and interaction_summary.validated_interactions > 0:
        trust_contributors.append(
            FeatureContributor("TRUST_VALIDATED_HISTORY", SupportSource.VALIDATED_HISTORY,
                               Provenance.STRUCTURED_METADATA)
        )
    trust_level = FeatureLevel.UNKNOWN if not trust_contributors else (
        FeatureLevel.HIGH if len(trust_contributors) >= 2 else FeatureLevel.MEDIUM
    )
    features.append(
        Feature(FeatureFamily.VALIDATED_TRUST, trust_level,
                tuple(sorted(trust_contributors, key=lambda c: c.sort_key())),
                detail=None)
    )

    # -- relevance --------------------------------------------------------
    relevance_contributors: list[FeatureContributor] = []
    conversation: ConversationContext = message.conversation
    if directly_addressed:
        relevance_contributors.append(
            FeatureContributor("REL_DIRECT_TARGET", SupportSource.STRUCTURED_METADATA,
                               Provenance.STRUCTURED_METADATA)
        )
    if conversation.surface.name == "DIRECT":
        relevance_contributors.append(
            FeatureContributor("REL_DIRECT_CONVERSATION", SupportSource.STRUCTURED_METADATA,
                               Provenance.STRUCTURED_METADATA)
        )
    if sender is not None and sender.has_asserted_trust:
        relevance_contributors.append(
            FeatureContributor("REL_VALIDATED_RELATIONSHIP", SupportSource.STRUCTURED_METADATA,
                               Provenance.STRUCTURED_METADATA)
        )
    if interaction_summary is not None and interaction_summary.validated_interactions > 0:
        relevance_contributors.append(
            FeatureContributor("REL_PRIOR_INTERACTION", SupportSource.VALIDATED_HISTORY,
                               Provenance.STRUCTURED_METADATA)
        )
    relevance_level = _tiered(len(relevance_contributors))
    if not relevance_contributors and conversation.surface.name == "CHANNEL":
        # A broadcast channel the reader was not addressed in is the one
        # case where "not relevant to this reader" was actually determined
        # rather than merely unavailable: surface and targeting are both
        # structured metadata, and both were consulted. Everywhere else,
        # missing context stays UNKNOWN.
        relevance_level = FeatureLevel.LOW
        relevance_contributors.append(
            FeatureContributor("REL_UNTARGETED_BROADCAST", SupportSource.STRUCTURED_METADATA,
                               Provenance.STRUCTURED_METADATA)
        )
    features.append(
        Feature(FeatureFamily.RECIPIENT_RELEVANCE, relevance_level,
                tuple(sorted(relevance_contributors, key=lambda c: c.sort_key())))
    )

    # -- utility ----------------------------------------------------------
    utility_contributors: list[FeatureContributor] = []
    if selected_name in _UTILITY_CLASSES:
        utility_contributors.append(
            FeatureContributor(f"UTIL_{selected_name}", SupportSource.UNTRUSTED_CONTENT,
                               Provenance.AUTHORED_TEXT)
        )
    utility_level = FeatureLevel.MEDIUM if utility_contributors else FeatureLevel.UNKNOWN
    if selected_name == "MARKETING":
        utility_level = FeatureLevel.LOW
    features.append(Feature(FeatureFamily.PRACTICAL_UTILITY, utility_level, tuple(utility_contributors)))

    # -- affinity ---------------------------------------------------------
    affinity_contributors: list[FeatureContributor] = []
    affinity_level = FeatureLevel.UNKNOWN
    if interaction_summary is None:
        affinity_contributors.append(
            FeatureContributor("AFFINITY_NO_SUMMARY", SupportSource.VALIDATED_HISTORY,
                               Provenance.STRUCTURED_METADATA, FeatureNote.NO_HISTORY_SUMMARY)
        )
    else:
        affinity_contributors.append(
            FeatureContributor("AFFINITY_VALIDATED_SUMMARY", SupportSource.VALIDATED_HISTORY,
                               Provenance.STRUCTURED_METADATA)
        )
        affinity_level = interaction_summary.affinity_level()
    features.append(Feature(FeatureFamily.RECIPIENT_AFFINITY, affinity_level, tuple(affinity_contributors)))

    # -- quiet hours / silenced thread / interruption cost ----------------
    quiet_level, quiet_notes = _quiet_hours(preferences, local_minute_of_day)
    features.append(Feature(FeatureFamily.QUIET_HOURS_IMPACT, quiet_level, quiet_notes))

    silenced_level = FeatureLevel.UNKNOWN
    silenced_notes: tuple[FeatureContributor, ...] = (
        FeatureContributor("SILENCED_NO_PREFERENCES", SupportSource.RECIPIENT_PREFERENCES,
                           Provenance.STRUCTURED_METADATA, FeatureNote.PREFERENCES_UNAVAILABLE),
    )
    if preferences is not None:
        silenced = preferences.has_muted(conversation.conversation_id)
        silenced_level = FeatureLevel.HIGH if silenced else FeatureLevel.ABSENT
        silenced_notes = (
            FeatureContributor("SILENCED_PREFERENCE", SupportSource.RECIPIENT_PREFERENCES,
                               Provenance.STRUCTURED_METADATA),
        )
    features.append(Feature(FeatureFamily.SILENCED_CONVERSATION, silenced_level, silenced_notes))

    interruption = _interruption_cost(quiet_level, silenced_level)
    features.append(
        Feature(FeatureFamily.INTERRUPTION_COST, interruption,
                (FeatureContributor("COST_FROM_PREFERENCES", SupportSource.RECIPIENT_PREFERENCES,
                                    Provenance.STRUCTURED_METADATA),))
    )

    # -- repetition -------------------------------------------------------
    repetition_contributors: list[FeatureContributor] = []
    repetition_level = FeatureLevel.UNKNOWN
    if forwarding_count is not None:
        repetition_level = FeatureLevel.HIGH if forwarding_count >= 5 else (
            FeatureLevel.LOW if forwarding_count > 0 else FeatureLevel.ABSENT
        )
        repetition_contributors.append(
            FeatureContributor("REPEAT_FORWARD_METADATA", SupportSource.STRUCTURED_METADATA,
                               Provenance.STRUCTURED_METADATA)
        )
    if any(f.category is FindingCategory.PROPAGATION_RISK for f in assessment.findings):
        repetition_contributors.append(
            FeatureContributor("REPEAT_PROPAGATION_FINDING", SupportSource.SAFETY_FINDINGS,
                               Provenance.DERIVED_SIGNAL, ambiguous=True)
        )
        if repetition_level in (FeatureLevel.UNKNOWN, FeatureLevel.ABSENT):
            repetition_level = FeatureLevel.LOW
    features.append(
        Feature(FeatureFamily.REPETITION, repetition_level,
                tuple(sorted(repetition_contributors, key=lambda c: c.sort_key())))
    )

    # -- modality / completeness / availability / ambiguity ---------------
    consistency, consistency_notes = _modality_consistency(assessment)
    features.append(
        Feature(FeatureFamily.MODALITY_CONSISTENCY, _consistency_level(consistency), consistency_notes,
                detail=consistency.value)
    )

    completeness = _completeness(assessment)
    features.append(
        Feature(
            FeatureFamily.ANALYSIS_COMPLETENESS,
            FeatureLevel.INCOMPLETE if assessment.analysis_incomplete else FeatureLevel.HIGH,
            (FeatureContributor("COMPLETENESS_PROJECTION", SupportSource.SAFETY_FINDINGS,
                                Provenance.DERIVED_SIGNAL,
                                FeatureNote.ANALYSIS_INCOMPLETE if assessment.analysis_incomplete else None,
                                analysis_unavailable=assessment.analysis_incomplete),),
            detail=completeness.value,
        )
    )

    availability = _content_availability(message, assessment, consistency)
    features.append(
        Feature(FeatureFamily.CONTENT_AVAILABILITY,
                FeatureLevel.ABSENT if availability is ContentAvailability.NO_USABLE_CONTENT else FeatureLevel.HIGH,
                (FeatureContributor("AVAILABILITY_PROJECTION", SupportSource.SAFETY_FINDINGS,
                                    Provenance.DERIVED_SIGNAL),),
                detail=availability.value)
    )

    ambiguous = (
        classification_ambiguous
        or consistency is ModalityConsistency.CONFLICTING
        or assessment.analysis_incomplete
        or any(f.ambiguous for f in assessment.findings)
    )
    features.append(
        Feature(FeatureFamily.AMBIGUITY,
                FeatureLevel.HIGH if ambiguous else FeatureLevel.ABSENT,
                (FeatureContributor("AMBIGUITY_PROJECTION", SupportSource.SAFETY_FINDINGS,
                                    Provenance.DERIVED_SIGNAL, ambiguous=ambiguous),))
    )

    root_causes = _risk_root_causes(assessment)
    return DecisionFeatures(
        features=tuple(sorted(features, key=lambda f: f.family.value)),
        highest_severity=severity,
        hard_veto_required=veto,
        risk_categories=assessment.categories,
        risk_families=_risk_families(root_causes),
        risk_root_causes=root_causes,
        cross_modal_corroboration=_cross_modal_corroboration(assessment),
        modality_consistency=consistency,
        completeness=completeness,
        content_availability=availability,
        urgency_outside_safety_coverage=urgency_outside_safety_coverage,
    )


def _legitimate_urgency(
    selected_name: str,
    disruption: FeatureContributor | None,
    near_term_event: bool,
    directly_addressed: bool,
    pressure_present: bool,
) -> tuple[FeatureLevel, list[FeatureContributor]]:
    """Legitimate urgency and its contributors. Never raised by a safety finding."""
    urgency_contributors: list[FeatureContributor] = []
    if selected_name in _URGENT_CLASSES:
        urgency_contributors.append(
            FeatureContributor(f"URGENCY_{selected_name}", SupportSource.UNTRUSTED_CONTENT,
                               Provenance.AUTHORED_TEXT)
        )
    if disruption is not None:
        # A service the reader depends on is down *right now*. Narrow by
        # construction: scheduled maintenance, a resolved incident, a
        # retrospective, a rumour, and an ordinary status update all fail
        # the classification rule that produces this.
        urgency_contributors.append(disruption)
    if near_term_event:
        urgency_contributors.append(
            FeatureContributor("URGENCY_NEAR_TERM_EVENT", SupportSource.STRUCTURED_METADATA,
                               Provenance.STRUCTURED_METADATA)
        )
    if directly_addressed and urgency_contributors:
        # Targeting amplifies urgency that already has an independent basis;
        # it never originates it. Otherwise a promotion addressed to one
        # person would become urgent purely by being addressed.
        urgency_contributors.append(
            FeatureContributor("URGENCY_DIRECT_TARGET", SupportSource.STRUCTURED_METADATA,
                               Provenance.STRUCTURED_METADATA)
        )
    if pressure_present:
        # Recorded so a reader can see pressure was noticed and deliberately
        # not counted as urgency.
        urgency_contributors.append(
            FeatureContributor("URGENCY_RISK_ATTRIBUTED", SupportSource.SAFETY_FINDINGS,
                               Provenance.DERIVED_SIGNAL,
                               FeatureNote.URGENCY_WITHHELD_RISK_ATTRIBUTED)
        )
    supporting = [c for c in urgency_contributors if c.note is not FeatureNote.URGENCY_WITHHELD_RISK_ATTRIBUTED]
    if not supporting:
        return FeatureLevel.ABSENT, urgency_contributors
    if (
        selected_name in _SELF_SUFFICIENT_URGENCY
        or disruption is not None
        or len(supporting) >= 2
    ):
        return FeatureLevel.HIGH, urgency_contributors
    return FeatureLevel.MEDIUM, urgency_contributors


@dataclass(frozen=True)
class InteractionSummary:
    """Validated summary of prior interaction, supplied by the caller.

    Counts only. No message text, and no retrieval or ranking happens here.
    """

    validated_interactions: int = 0
    replies: int = 0
    dismissals: int = 0

    def __post_init__(self) -> None:
        for name in ("validated_interactions", "replies", "dismissals"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(ValidationIssue.OUT_OF_RANGE, name)

    def affinity_level(self) -> FeatureLevel:
        if self.validated_interactions == 0:
            return FeatureLevel.UNKNOWN
        if self.replies > 0 and self.replies >= self.dismissals:
            return FeatureLevel.HIGH
        if self.dismissals > self.replies:
            return FeatureLevel.LOW
        return FeatureLevel.MEDIUM


def _active_disruption_contributor(
    classification, within_safety_coverage: bool = False
) -> FeatureContributor | None:
    """Urgency support from a currently active, recorded service disruption.

    Read by rule identity from the classifier's own contributors, so no
    content is inspected here and the provenance the signal came from is
    carried through unchanged. With ``within_safety_coverage``, a disruption
    found only where the safety layer did not read is not counted.
    """
    for contributor in getattr(classification, "contributors", ()):
        if within_safety_coverage and getattr(contributor, "outside_safety_coverage", False):
            continue
        if getattr(contributor, "rule_id", None) == ACTIVE_DISRUPTION_RULE_ID:
            return FeatureContributor(
                "URGENCY_ACTIVE_DISRUPTION",
                SupportSource.UNTRUSTED_CONTENT,
                contributor.provenance,
            )
    return None


def _tiered(count: int) -> FeatureLevel:
    if count == 0:
        return FeatureLevel.UNKNOWN
    if count == 1:
        return FeatureLevel.MEDIUM
    return FeatureLevel.HIGH


def _quiet_hours(
    preferences: RecipientPreferences | None, local_minute_of_day: int | None
) -> tuple[FeatureLevel, tuple[FeatureContributor, ...]]:
    if preferences is None or preferences.quiet_period is None:
        return FeatureLevel.UNKNOWN, (
            FeatureContributor("QUIET_NO_PREFERENCE", SupportSource.RECIPIENT_PREFERENCES,
                               Provenance.STRUCTURED_METADATA, FeatureNote.PREFERENCES_UNAVAILABLE),
        )
    if local_minute_of_day is None:
        return FeatureLevel.UNKNOWN, (
            FeatureContributor("QUIET_LOCAL_TIME_UNKNOWN", SupportSource.RECIPIENT_PREFERENCES,
                               Provenance.STRUCTURED_METADATA, FeatureNote.QUIET_HOURS_UNKNOWN),
        )
    inside = preferences.quiet_period.contains_minute(local_minute_of_day)
    return (FeatureLevel.HIGH if inside else FeatureLevel.ABSENT), (
        FeatureContributor("QUIET_WINDOW_EVALUATED", SupportSource.RECIPIENT_PREFERENCES,
                           Provenance.STRUCTURED_METADATA),
    )


def _interruption_cost(quiet: FeatureLevel, silenced: FeatureLevel) -> FeatureLevel:
    """Cost of interrupting, independent of whether interrupting is right.

    A perfectly safe, non-urgent message can carry a high cost. This says
    nothing about the eventual route.
    """
    if quiet is FeatureLevel.HIGH and silenced is FeatureLevel.HIGH:
        return FeatureLevel.HIGH
    if FeatureLevel.HIGH in (quiet, silenced):
        return FeatureLevel.MEDIUM
    if quiet is FeatureLevel.UNKNOWN and silenced is FeatureLevel.UNKNOWN:
        return FeatureLevel.UNKNOWN
    return FeatureLevel.LOW


def _modality_consistency(
    assessment: SafetyAssessment,
) -> tuple[ModalityConsistency, tuple[FeatureContributor, ...]]:
    notes: list[FeatureContributor] = []
    states = {o.state for o in assessment.modality_outcomes}

    for conflict in assessment.modality_conflicts:
        if conflict.kind is ConflictKind.INSUFFICIENT_COMPARISON:
            notes.append(
                FeatureContributor("MODALITY_INSUFFICIENT", SupportSource.SAFETY_FINDINGS,
                                   conflict.left, FeatureNote.INSUFFICIENT_COMPARISON, ambiguous=True)
            )
        else:
            notes.append(
                FeatureContributor(f"MODALITY_{conflict.kind.name}", SupportSource.SAFETY_FINDINGS,
                                   conflict.left, FeatureNote.MODALITY_CONFLICT, ambiguous=True)
            )

    real_conflicts = [
        c for c in assessment.modality_conflicts if c.kind is not ConflictKind.INSUFFICIENT_COMPARISON
    ]
    if real_conflicts:
        return ModalityConsistency.CONFLICTING, tuple(sorted(notes, key=lambda c: c.sort_key()))

    # Degradation is reported, and is never read as evidence of safety.
    for state, outcome in (
        (ModalityState.FAILED, ModalityConsistency.FAILED),
        (ModalityState.UNSUPPORTED, ModalityConsistency.UNSUPPORTED),
        (ModalityState.UNAVAILABLE, ModalityConsistency.UNAVAILABLE),
    ):
        if state in states:
            notes.append(
                FeatureContributor(f"MODALITY_{state.name}", SupportSource.SAFETY_FINDINGS,
                                   Provenance.DERIVED_SIGNAL, FeatureNote.MEDIA_ANALYSIS_DEGRADED,
                                   analysis_unavailable=True)
            )
            return outcome, tuple(sorted(notes, key=lambda c: c.sort_key()))

    if assessment.modality_conflicts:
        return ModalityConsistency.INSUFFICIENT, tuple(sorted(notes, key=lambda c: c.sort_key()))

    analyzed = [o for o in assessment.modality_outcomes if o.state is ModalityState.ANALYZED]
    if len(analyzed) >= 2:
        return ModalityConsistency.CONSISTENT, tuple(notes)
    notes.append(
        FeatureContributor("MODALITY_SINGLE_SOURCE", SupportSource.SAFETY_FINDINGS,
                           Provenance.DERIVED_SIGNAL, FeatureNote.NO_SECONDARY_MODALITY)
    )
    return ModalityConsistency.NOT_APPLICABLE, tuple(notes)


def _consistency_level(consistency: ModalityConsistency) -> FeatureLevel:
    if consistency is ModalityConsistency.CONSISTENT:
        return FeatureLevel.HIGH
    if consistency is ModalityConsistency.CONFLICTING:
        return FeatureLevel.CONFLICTED
    if consistency is ModalityConsistency.NOT_APPLICABLE:
        return FeatureLevel.ABSENT
    return FeatureLevel.INCOMPLETE


def _completeness(assessment: SafetyAssessment) -> Completeness:
    states = {o.state for o in assessment.modality_outcomes}
    if any(c.kind is not ConflictKind.INSUFFICIENT_COMPARISON for c in assessment.modality_conflicts):
        return Completeness.CONFLICTED
    if ModalityState.FAILED in states:
        return Completeness.INCOMPLETE_FAILED_ANALYSIS
    if ModalityState.UNSUPPORTED in states:
        return Completeness.INCOMPLETE_UNSUPPORTED_MEDIA
    if ModalityState.UNAVAILABLE in states:
        return Completeness.INCOMPLETE_UNAVAILABLE_BACKEND
    analyzed = any(o.state is ModalityState.ANALYZED for o in assessment.modality_outcomes)
    if not analyzed:
        return Completeness.INSUFFICIENT_CONTENT
    if assessment.analysis_incomplete:
        return Completeness.PARTIALLY_COMPLETE
    return Completeness.COMPLETE_FOR_AVAILABLE_INPUTS


def _content_availability(
    message: IncomingMessage, assessment: SafetyAssessment, consistency: ModalityConsistency
) -> ContentAvailability:
    if consistency is ModalityConsistency.CONFLICTING:
        return ContentAvailability.CONFLICTED_CONTENT
    authored = any(
        o.provenance is Provenance.AUTHORED_TEXT and o.state is ModalityState.ANALYZED
        for o in assessment.modality_outcomes
    )
    extracted = any(
        o.provenance is not Provenance.AUTHORED_TEXT and o.state is ModalityState.ANALYZED
        for o in assessment.modality_outcomes
    )
    if authored and extracted:
        return ContentAvailability.AUTHORED_AND_EXTRACTED
    if authored:
        return ContentAvailability.AUTHORED_ONLY
    if extracted:
        return ContentAvailability.EXTRACTED_ONLY
    if message.sender is not None or message.attachment is not None:
        return ContentAvailability.STRUCTURED_METADATA_ONLY
    return ContentAvailability.NO_USABLE_CONTENT
