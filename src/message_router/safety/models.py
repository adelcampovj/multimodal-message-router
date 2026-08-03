"""Immutable results of safety analysis.

A finding says *what was observed and how strongly*, never *what to do
about it*. Nothing in this module carries a routing action, a message
category, a rationale, a confidence number, or an evidence score -- those
are decided later, by a component that reads these findings.

Nothing here stores content either. A finding carries a rule identifier, a
category, a provenance, a sanitized source reference, and a fixed
diagnostic code. It never carries the text that triggered it, the secret it
noticed being requested, the URL it found suspicious, or the payload it was
decoded from.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..domain.enums import Provenance, Reliability

__all__ = [
    "FindingCategory",
    "Severity",
    "DiagnosticCode",
    "ModalityState",
    "ConflictKind",
    "SafetyFinding",
    "ModalityOutcome",
    "ModalityConflict",
    "SafetyAssessment",
    "SOURCE_CEILING",
    "AUTHORED_SOURCE_REF",
    "STRUCTURED_SOURCE_REF",
]

#: Sanitized stand-ins used where a finding must name where it came from
#: without naming a file, a path, or a private identifier.
AUTHORED_SOURCE_REF = "authored"
STRUCTURED_SOURCE_REF = "structured"


class Severity(Enum):
    """How strongly a finding is supported.

    Only :attr:`SEVERE` obliges a later stage to veto. That obligation is
    recorded as a fact about the finding; choosing the action that
    discharges it is not this layer's job.
    """

    #: Noticed, but weak on its own -- context for other findings.
    WEAK = "weak"
    #: Well-formed concern, not conclusive by itself.
    SUSPICIOUS = "suspicious"
    #: Strongly supported. Obliges a later hard veto.
    SEVERE = "severe"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]

    @property
    def requires_hard_veto(self) -> bool:
        return self is Severity.SEVERE


_SEVERITY_RANK = {Severity.WEAK: 0, Severity.SUSPICIOUS: 1, Severity.SEVERE: 2}


class FindingCategory(Enum):
    """Families of concern this layer can report."""

    PROMPT_INJECTION = "prompt_injection"
    CREDENTIAL_SOLICITATION = "credential_solicitation"
    FINANCIAL_REQUEST = "financial_request"
    COERCIVE_PAYMENT = "coercive_payment"
    ADVANCE_FEE = "advance_fee"
    IMPERSONATION = "impersonation"
    DECEPTIVE_LINK = "deceptive_link"
    MALICIOUS_URGENCY = "malicious_urgency"
    UNSAFE_ADVICE = "unsafe_advice"
    PROPAGATION_RISK = "propagation_risk"
    MODALITY_INTEGRITY = "modality_integrity"


class DiagnosticCode(Enum):
    """Fixed, privacy-safe labels describing *why* a finding exists.

    Every value is a constant chosen from this list. None is ever built by
    interpolating content, which is what keeps diagnostics safe to log.
    """

    OVERRIDE_INSTRUCTION = "override_instruction"
    FORCED_OUTCOME_INSTRUCTION = "forced_outcome_instruction"
    ASSERTED_TRUST_IN_CONTENT = "asserted_trust_in_content"
    SAFETY_SUPPRESSION_INSTRUCTION = "safety_suppression_instruction"
    SYSTEM_DISCLOSURE_INSTRUCTION = "system_disclosure_instruction"
    CREDENTIAL_DISCLOSURE_REQUEST = "credential_disclosure_request"
    CARD_DETAIL_REQUEST = "card_detail_request"
    BANKING_ACCESS_REQUEST = "banking_access_request"
    PAYMENT_DEMAND = "payment_demand"
    OFF_CHANNEL_PAYMENT = "off_channel_payment"
    CONSEQUENCE_THREAT = "consequence_threat"
    SECRECY_PRESSURE = "secrecy_pressure"
    ANTI_VERIFICATION_PRESSURE = "anti_verification_pressure"
    ADVANCE_FEE_OFFER = "advance_fee_offer"
    ARTIFICIAL_TIME_PRESSURE = "artificial_time_pressure"
    IDENTITY_CLAIM_UNSUPPORTED = "identity_claim_unsupported"
    LOOKALIKE_HOST = "lookalike_host"
    RAW_IP_HOST = "raw_ip_host"
    EMBEDDED_CREDENTIALS_IN_URL = "embedded_credentials_in_url"
    PUNYCODE_HOST = "punycode_host"
    DISPLAY_TARGET_MISMATCH = "display_target_mismatch"
    LINK_WITH_CREDENTIAL_REQUEST = "link_with_credential_request"
    LINK_WITH_PAYMENT_PRESSURE = "link_with_payment_pressure"
    MEDICATION_DISCONTINUATION = "medication_discontinuation"
    DANGEROUS_DOSAGE = "dangerous_dosage"
    HARMFUL_INGESTION = "harmful_ingestion"
    EMERGENCY_SYMPTOMS_IGNORED = "emergency_symptoms_ignored"
    EMERGENCY_CARE_DISCOURAGED = "emergency_care_discouraged"
    CONCEALMENT_FROM_CARE = "concealment_from_care"
    HIGH_FORWARD_COUNT = "high_forward_count"
    CHAIN_MESSAGE_LANGUAGE = "chain_message_language"
    RESEND_PRESSURE = "resend_pressure"
    UNVERIFIABLE_MASS_WARNING = "unverifiable_mass_warning"
    AUTHORED_EXTRACTED_CONFLICT = "authored_extracted_conflict"
    EXTRACTED_EXTRACTED_CONFLICT = "extracted_extracted_conflict"
    INSUFFICIENT_COMPARISON = "insufficient_comparison"
    ANALYSIS_INCOMPLETE = "analysis_incomplete"
    CONTENT_TRUNCATED = "content_truncated"
    NO_AUTHORED_BODY = "no_authored_body"
    SUPPRESSED_BY_WARNING_CONTEXT = "suppressed_by_warning_context"
    QUOTED_OR_EDUCATIONAL_CONTEXT = "quoted_or_educational_context"


class ModalityState(Enum):
    """What happened when a source was consulted.

    Kept separate from severity: none of these is itself a safety verdict,
    and in particular none of them is evidence that a message is safe.
    """

    NOT_APPLICABLE = "not_applicable"
    ANALYZED = "analyzed"
    EMPTY = "empty"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"

    @property
    def yielded_content(self) -> bool:
        return self is ModalityState.ANALYZED

    @property
    def is_degraded(self) -> bool:
        """True when a source could have contributed but did not.

        ``NOT_APPLICABLE`` is excluded: an absent attachment is not a gap
        in analysis, it is simply nothing to analyse.
        """
        return self in _DEGRADED_STATES


_DEGRADED_STATES = frozenset(
    {ModalityState.UNAVAILABLE, ModalityState.UNSUPPORTED, ModalityState.FAILED}
)


class ConflictKind(Enum):
    """How two sources disagreed."""

    AUTHORED_VS_EXTRACTED = "authored_vs_extracted"
    EXTRACTED_VS_EXTRACTED = "extracted_vs_extracted"
    INSUFFICIENT_COMPARISON = "insufficient_comparison"


#: The most reliability a finding drawn from each source may claim.
#: Authored text is what a person actually wrote; everything else has been
#: through a lossy, attacker-influenceable step and cannot outrank it.
SOURCE_CEILING = {
    Provenance.STRUCTURED_METADATA: Reliability.HIGH,
    Provenance.AUTHORED_TEXT: Reliability.HIGH,
    Provenance.OCR_TEXT: Reliability.MEDIUM,
    Provenance.ASR_TRANSCRIPT: Reliability.MEDIUM,
    Provenance.QR_DERIVED: Reliability.MEDIUM,
    Provenance.URL_DERIVED: Reliability.LOW,
    Provenance.MODEL_INTERPRETATION: Reliability.LOW,
    Provenance.DERIVED_SIGNAL: Reliability.MEDIUM,
}

_RELIABILITY_RANK = {Reliability.LOW: 0, Reliability.MEDIUM: 1, Reliability.HIGH: 2}
_PROVENANCE_ORDER = {member: index for index, member in enumerate(Provenance)}


@dataclass(frozen=True)
class SafetyFinding:
    """One observation, with no content attached."""

    category: FindingCategory
    severity: Severity
    rule_id: str
    provenance: Provenance
    source_ref: str
    diagnostic: DiagnosticCode
    reliability: Reliability = Reliability.MEDIUM
    ambiguous: bool = False
    indicators: tuple[str, ...] = ()

    @property
    def requires_hard_veto(self) -> bool:
        """Whether a later stage is obliged to veto because of this finding."""
        return self.severity.requires_hard_veto

    def sort_key(self) -> tuple:
        """Total order: strongest first, then stable and content-independent."""
        return (
            -self.severity.rank,
            self.category.value,
            self.rule_id,
            _PROVENANCE_ORDER.get(self.provenance, 99),
            self.source_ref,
            self.diagnostic.value,
        )


@dataclass(frozen=True)
class ModalityOutcome:
    """What one source contributed, and how far it can be relied on."""

    provenance: Provenance
    source_ref: str
    state: ModalityState
    reliability: Reliability = Reliability.MEDIUM

    def sort_key(self) -> tuple:
        return (_PROVENANCE_ORDER.get(self.provenance, 99), self.source_ref, self.state.value)


@dataclass(frozen=True)
class ModalityConflict:
    """A disagreement between two sources about the same message."""

    kind: ConflictKind
    left: Provenance
    right: Provenance
    diagnostic: DiagnosticCode

    def sort_key(self) -> tuple:
        return (
            self.kind.value,
            _PROVENANCE_ORDER.get(self.left, 99),
            _PROVENANCE_ORDER.get(self.right, 99),
        )


@dataclass(frozen=True)
class SafetyAssessment:
    """Everything the safety layer concluded about one message.

    Deliberately absent, and asserted absent by a structural test: routing
    action, message category, rationale, confidence, evidence ranking, and
    any preference or engagement adjustment. This object describes risk; it
    does not decide delivery.
    """

    findings: tuple[SafetyFinding, ...] = ()
    modality_outcomes: tuple[ModalityOutcome, ...] = ()
    modality_conflicts: tuple[ModalityConflict, ...] = ()
    diagnostics: tuple[DiagnosticCode, ...] = ()
    analysis_incomplete: bool = False

    @property
    def highest_severity(self) -> Severity | None:
        """The strongest severity present, or None when nothing was found."""
        if not self.findings:
            return None
        return max((finding.severity for finding in self.findings), key=lambda s: s.rank)

    @property
    def hard_veto_required(self) -> bool:
        """True when any finding obliges a later veto.

        Computed from the findings themselves, so it cannot drift away from
        them, and cannot be set directly by anything -- least of all by
        content claiming it should be false.
        """
        return any(finding.requires_hard_veto for finding in self.findings)

    @property
    def categories(self) -> frozenset[FindingCategory]:
        return frozenset(finding.category for finding in self.findings)

    @property
    def degraded_sources(self) -> tuple[ModalityOutcome, ...]:
        """Sources that could have contributed but did not.

        Their presence is a reason to be *less* certain, never a reason to
        conclude a message is safe.
        """
        return tuple(outcome for outcome in self.modality_outcomes if outcome.state.is_degraded)


def cap_reliability(reliability: Reliability, provenance: Provenance) -> Reliability:
    """Lower ``reliability`` to whatever ``provenance`` can support."""
    ceiling = SOURCE_CEILING.get(provenance, Reliability.LOW)
    if _RELIABILITY_RANK[reliability] <= _RELIABILITY_RANK[ceiling]:
        return reliability
    return ceiling
