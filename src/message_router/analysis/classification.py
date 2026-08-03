"""What a message *is*, decided independently of what to do about it.

Classification describes meaning. Routing decides delivery. They are
separate types here and neither is derived from the other: the classifier
never sees a routing action, and nothing in this module returns one.

That separation matters in both directions. A message can be
:attr:`~MessageClass.SUSPICIOUS_CONTENT` without that alone meaning it must
be withheld; it can be a :attr:`~MessageClass.FINANCIAL_NOTICE` without
being risky; it can be a :attr:`~MessageClass.DIRECT_REQUEST` without being
urgent. Those are routing judgements, made later, with more inputs.

Rules here are conservative pattern families. They make no claim to
semantic understanding: when support is weak or contributors disagree, the
result stays ambiguous or falls to
:attr:`~MessageClass.UNCATEGORIZED` rather than manufacturing precision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..domain.enums import Provenance
from ..domain.errors import UnsupportedValueError
from ..safety.models import FindingCategory, SafetyAssessment, Severity

__all__ = [
    "MessageClass",
    "SupportStrength",
    "ClassificationDiagnostic",
    "ClassificationContributor",
    "ClassificationResult",
    "classify",
]


class MessageClass(Enum):
    """What a message is about.

    Extensible by addition. Unknown values are never absorbed silently --
    :meth:`from_value` raises rather than falling back to
    :attr:`UNCATEGORIZED`, so a typo cannot quietly become a default.
    """

    #: Someone is asking this recipient to do or answer something.
    DIRECT_REQUEST = "direct_request"
    #: A warning about danger, security, or an emergency.
    SAFETY_ALERT = "safety_alert"
    #: An appointment, booking, meeting, or dated commitment.
    SCHEDULED_EVENT = "scheduled_event"
    #: An invoice, receipt, charge, refund, or payment record.
    FINANCIAL_NOTICE = "financial_notice"
    #: Delivery, order, account, or service status.
    SERVICE_UPDATE = "service_update"
    #: A promotion, offer, or campaign.
    MARKETING = "marketing"
    #: Ordinary conversation between people.
    SOCIAL_MESSAGE = "social_message"
    #: Content forwarded or circulated rather than written for this reader.
    PROPAGATED_CONTENT = "propagated_content"
    #: Content the safety layer raised concerns about.
    SUSPICIOUS_CONTENT = "suspicious_content"
    #: Deliberately unassigned. Never a silent fallback for a parse failure.
    UNCATEGORIZED = "uncategorized"

    @classmethod
    def from_value(cls, raw: object, field: str = "message_class") -> "MessageClass":
        for member in cls:
            if member.value == raw:
                return member
        raise UnsupportedValueError(field)


class SupportStrength(Enum):
    """How firmly one contributor backs its candidate."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"

    @property
    def rank(self) -> int:
        return _STRENGTH_RANK[self]


_STRENGTH_RANK = {SupportStrength.WEAK: 0, SupportStrength.MODERATE: 1, SupportStrength.STRONG: 2}


class ClassificationDiagnostic(Enum):
    """Fixed, privacy-safe labels explaining a contributor or a tie-break."""

    REQUEST_LANGUAGE = "request_language"
    DIRECT_ADDRESS = "direct_address"
    WARNING_LANGUAGE = "warning_language"
    SCHEDULING_LANGUAGE = "scheduling_language"
    BILLING_LANGUAGE = "billing_language"
    STATUS_LANGUAGE = "status_language"
    ACTIVE_DISRUPTION_LANGUAGE = "active_disruption_language"
    PROMOTIONAL_LANGUAGE = "promotional_language"
    CONVERSATIONAL_LANGUAGE = "conversational_language"
    FORWARDING_LANGUAGE = "forwarding_language"
    FORWARDING_METADATA = "forwarding_metadata"
    SAFETY_FINDING_PRESENT = "safety_finding_present"
    SEVERE_SAFETY_FINDING = "severe_safety_finding"
    STRUCTURED_SURFACE = "structured_surface"
    NO_USABLE_CONTENT = "no_usable_content"
    CONTRIBUTORS_DISAGREE = "contributors_disagree"
    NO_CONTRIBUTOR_REACHED_THRESHOLD = "no_contributor_reached_threshold"
    RESOLVED_BY_STRENGTH = "resolved_by_strength"
    RESOLVED_BY_STABLE_ORDER = "resolved_by_stable_order"
    ANALYSIS_INCOMPLETE = "analysis_incomplete"


@dataclass(frozen=True)
class ClassificationContributor:
    """One reason to believe a message belongs to a class.

    Carries no content: a rule identifier, a fixed diagnostic, and the
    provenance the support came from.
    """

    candidate: MessageClass
    provenance: Provenance
    rule_id: str
    strength: SupportStrength
    diagnostic: ClassificationDiagnostic
    ambiguous: bool = False
    from_incomplete_analysis: bool = False

    def sort_key(self) -> tuple:
        return (
            -self.strength.rank,
            self.candidate.value,
            self.rule_id,
            _PROVENANCE_ORDER.get(self.provenance, 99),
            self.diagnostic.value,
        )


_PROVENANCE_ORDER = {member: index for index, member in enumerate(Provenance)}


@dataclass(frozen=True)
class ClassificationResult:
    """The selected class plus every reason considered.

    Deliberately absent, and asserted absent by a structural test: routing
    action, route score, confidence, rationale, and evidence ranking.
    """

    selected: MessageClass
    contributors: tuple[ClassificationContributor, ...] = ()
    ambiguous: bool = False
    analysis_incomplete: bool = False
    tie_break: ClassificationDiagnostic | None = None

    @property
    def considered(self) -> frozenset[MessageClass]:
        return frozenset(c.candidate for c in self.contributors)


# ---------------------------------------------------------------------------
# Conservative pattern families.
# ---------------------------------------------------------------------------


def _c(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_PATTERNS: tuple[tuple[str, MessageClass, SupportStrength, ClassificationDiagnostic, re.Pattern[str]], ...] = (
    (
        "CLS_REQUEST",
        MessageClass.DIRECT_REQUEST,
        SupportStrength.MODERATE,
        ClassificationDiagnostic.REQUEST_LANGUAGE,
        _c(r"\b(?:can|could|would|will) you\b|\bplease (?:send|reply|confirm|let me know|bring|call|check)\b"
           r"|\bcould you please\b|\blet me know\b|\bwhat time\b|\bare you able to\b|\bneed you to\b"),
    ),
    (
        "CLS_SAFETY_ALERT",
        MessageClass.SAFETY_ALERT,
        SupportStrength.STRONG,
        ClassificationDiagnostic.WARNING_LANGUAGE,
        _c(r"\b(?:evacuat|fire alarm|gas leak|flood warning|severe weather|earthquake|lockdown)\w*\b"
           r"|\b(?:safety|emergency|weather) (?:alert|warning|notice)\b|\bseek shelter\b|\bdo not enter\b"),
    ),
    (
        "CLS_SCHEDULED",
        MessageClass.SCHEDULED_EVENT,
        SupportStrength.MODERATE,
        ClassificationDiagnostic.SCHEDULING_LANGUAGE,
        _c(r"\bappointment\b|\bbooking\b|\breservation\b|\bmeeting\b|\bscheduled for\b|\bcalendar\b"
           r"|\brescheduled?\b|\bon (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
           r"|\bat \d{1,2}(?::\d{2})? ?(?:am|pm)\b"),
    ),
    (
        "CLS_FINANCIAL",
        MessageClass.FINANCIAL_NOTICE,
        SupportStrength.MODERATE,
        ClassificationDiagnostic.BILLING_LANGUAGE,
        _c(r"\binvoice\b|\breceipt\b|\bstatement\b|\brefund\b|\bcharged?\b|\bpayment (?:received|confirmed|due)\b"
           r"|\bbalance\b|\bdirect debit\b|\bsubscription (?:renew|charge)\w*\b|\btransaction\b"),
    ),
    (
        "CLS_SERVICE",
        MessageClass.SERVICE_UPDATE,
        SupportStrength.MODERATE,
        ClassificationDiagnostic.STATUS_LANGUAGE,
        _c(r"\b(?:order|parcel|package|delivery|shipment) (?:is|has|will|status|update)\b"
           r"|\bout for delivery\b|\bhas shipped\b|\btracking (?:number|update)\b"
           r"|\byour account (?:has|was|is) (?:updated|created|changed)\b|\bservice (?:status|disruption|restored)\b"
           r"|\bmaintenance window\b"),
    ),
    (
        "CLS_MARKETING",
        MessageClass.MARKETING,
        SupportStrength.MODERATE,
        ClassificationDiagnostic.PROMOTIONAL_LANGUAGE,
        _c(r"\b\d{1,3}% off\b|\bsale\b|\bdiscount\b|\boffer ends\b|\blimited time offer\b|\bshop now\b"
           r"|\bunsubscribe\b|\bnew collection\b|\bdeal of the\b|\bpromo code\b"),
    ),
    (
        "CLS_SOCIAL",
        MessageClass.SOCIAL_MESSAGE,
        SupportStrength.WEAK,
        ClassificationDiagnostic.CONVERSATIONAL_LANGUAGE,
        _c(r"\b(?:good morning|good evening|good night|happy birthday|congratulations|congrats)\b"
           r"|\bhow are you\b|\bhope you(?:'re| are) well\b|\bthanks?(?: so much)?\b|\bsee you\b|\bmiss you\b"),
    ),
    (
        "CLS_PROPAGATED",
        MessageClass.PROPAGATED_CONTENT,
        SupportStrength.MODERATE,
        ClassificationDiagnostic.FORWARDING_LANGUAGE,
        _c(r"\bforward(?:ed)? (?:this|message|to)\b|\bshare (?:this )?with (?:everyone|all|\d+)\b"
           r"|\bpass (?:this|it) on\b|\bcopy and paste\b|\bdo ?n[o']?t break the chain\b|\bspread the word\b"),
    ),
)

#: Forwarding metadata at or above this is a contributor on its own.
FORWARD_METADATA_THRESHOLD = 5

#: Rule identifier for a *currently active* service disruption. Named as a
#: constant because a later stage reads it by identity rather than by
#: re-inspecting content: this is the only classification signal that a
#: feature may treat as an independent basis for urgency.
ACTIVE_DISRUPTION_RULE_ID = "CLS_ACTIVE_DISRUPTION"

# Deliberately narrow. An ordinary service update, a maintenance notice, a
# resolved incident, a retrospective, a rumour, and a promotion must all
# fail this, because the only thing it is allowed to support downstream is
# treating something as urgent right now.
_ACTIVE_DISRUPTION_RE = _c(
    r"\b(?:service|system|platform|network|site|website|app|portal|server|api)\b[^.]{0,25}"
    r"\b(?:is|are|currently)\b[^.]{0,20}"
    r"\b(?:down|unavailable|inaccessible|offline|interrupted|disrupted|suspended)\b"
    r"|\b(?:ongoing|active|current) (?:outage|disruption|incident|downtime)\b"
    r"|\boutage (?:is )?(?:ongoing|in progress|affecting)\b"
    r"|\bwe are (?:currently )?(?:experiencing|investigating)\b[^.]{0,30}"
    r"\b(?:outage|disruption|incident|downtime|problems?)\b"
)

_DISRUPTION_EXCLUSIONS = tuple(
    _c(pattern)
    for pattern in (
        # Scheduled or future work is not an active disruption.
        r"\b(?:scheduled|planned|upcoming|forthcoming|due)\b[^.]{0,35}"
        r"\b(?:maintenance|outage|downtime|disruption|window)\b",
        r"\bmaintenance (?:window|is scheduled|will|begins)\b",
        r"\b(?:next|last|this coming) (?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        # Already over.
        r"\b(?:resolved|restored|fixed|back online|back up|no longer affected)\b",
        # Retrospective or documentation.
        r"\b(?:yesterday|last night|last year|back in|previously|used to be|in 20\d\d)\b",
        r"\b(?:this|the) (?:article|guide|post|documentation|docs|manual|runbook|report)\b",
        # Unconfirmed.
        r"\b(?:might|may|could|possibly|rumou?r|reportedly|apparently|i think|not sure|heard that)\b",
        # A promotion that happens to mention a service.
        r"\b(?:sale|discount|shop now|promo code|offer ends|unsubscribe)\b",
    )
)


def _is_active_disruption(raw: str) -> bool:
    folded = raw.casefold()
    if not _ACTIVE_DISRUPTION_RE.search(folded):
        return False
    return not any(pattern.search(folded) for pattern in _DISRUPTION_EXCLUSIONS)


def classify(
    texts: tuple[tuple[str, Provenance], ...] = (),
    assessment: SafetyAssessment | None = None,
    forwarding_count: int | None = None,
    directly_addressed: bool = False,
) -> ClassificationResult:
    """Classify a message from provenance-tagged content and safety findings.

    ``texts`` pairs each readable piece of content with the provenance it
    came from; sources are never concatenated, so a contributor always
    names the source that produced it.

    Accepts no routing action and returns none.
    """
    contributors: list[ClassificationContributor] = []
    incomplete = bool(assessment.analysis_incomplete) if assessment is not None else False

    for raw, provenance in texts:
        for rule_id, candidate, strength, diagnostic, pattern in _PATTERNS:
            if pattern.search(raw):
                contributors.append(
                    ClassificationContributor(
                        candidate=candidate,
                        provenance=provenance,
                        rule_id=rule_id,
                        strength=strength,
                        diagnostic=diagnostic,
                        from_incomplete_analysis=incomplete,
                    )
                )
        if _is_active_disruption(raw):
            contributors.append(
                ClassificationContributor(
                    candidate=MessageClass.SERVICE_UPDATE,
                    provenance=provenance,
                    rule_id=ACTIVE_DISRUPTION_RULE_ID,
                    strength=SupportStrength.STRONG,
                    diagnostic=ClassificationDiagnostic.ACTIVE_DISRUPTION_LANGUAGE,
                    from_incomplete_analysis=incomplete,
                )
            )

    # Direct addressing reinforces a request the content already contains;
    # it never creates one. Being addressed is not the same as being asked
    # something, and treating it as such would let routing metadata alone
    # manufacture a request out of an ordinary remark.
    if directly_addressed and any(c.candidate is MessageClass.DIRECT_REQUEST for c in contributors):
        contributors.append(
            ClassificationContributor(
                MessageClass.DIRECT_REQUEST,
                Provenance.STRUCTURED_METADATA,
                "CLS_DIRECT_ADDRESS",
                SupportStrength.WEAK,
                ClassificationDiagnostic.DIRECT_ADDRESS,
                from_incomplete_analysis=incomplete,
            )
        )

    if forwarding_count is not None and forwarding_count >= FORWARD_METADATA_THRESHOLD:
        contributors.append(
            ClassificationContributor(
                MessageClass.PROPAGATED_CONTENT,
                Provenance.STRUCTURED_METADATA,
                "CLS_FORWARD_METADATA",
                SupportStrength.WEAK,
                ClassificationDiagnostic.FORWARDING_METADATA,
                from_incomplete_analysis=incomplete,
            )
        )

    if assessment is not None and assessment.findings:
        severe = assessment.highest_severity is Severity.SEVERE
        # A safety finding contributes to what the message *is*. It does not
        # change the finding's own severity, and it does not decide delivery.
        risk_categories = assessment.categories - {FindingCategory.MODALITY_INTEGRITY}
        if risk_categories:
            contributors.append(
                ClassificationContributor(
                    MessageClass.SUSPICIOUS_CONTENT,
                    Provenance.DERIVED_SIGNAL,
                    "CLS_SAFETY_FINDING",
                    SupportStrength.STRONG if severe else SupportStrength.MODERATE,
                    ClassificationDiagnostic.SEVERE_SAFETY_FINDING
                    if severe
                    else ClassificationDiagnostic.SAFETY_FINDING_PRESENT,
                    ambiguous=not severe,
                    from_incomplete_analysis=incomplete,
                )
            )

    ordered = tuple(sorted(contributors, key=lambda c: c.sort_key()))
    selected, ambiguous, tie_break = _select(ordered, bool(texts))
    return ClassificationResult(
        selected=selected,
        contributors=ordered,
        ambiguous=ambiguous,
        analysis_incomplete=incomplete,
        tie_break=tie_break,
    )


def _select(
    contributors: tuple[ClassificationContributor, ...], had_content: bool
) -> tuple[MessageClass, bool, ClassificationDiagnostic | None]:
    """Pick a class deterministically, preserving ambiguity honestly."""
    if not contributors:
        return (
            MessageClass.UNCATEGORIZED,
            False,
            ClassificationDiagnostic.NO_CONTRIBUTOR_REACHED_THRESHOLD
            if had_content
            else ClassificationDiagnostic.NO_USABLE_CONTENT,
        )

    totals: dict[MessageClass, int] = {}
    for contributor in contributors:
        totals[contributor.candidate] = totals.get(contributor.candidate, 0) + contributor.strength.rank + 1

    best = max(totals.values())
    leaders = sorted((c for c, total in totals.items() if total == best), key=lambda c: c.value)
    distinct_candidates = len(totals) > 1

    if len(leaders) > 1:
        # A genuine tie is reported as ambiguous and resolved by a stable
        # rule, never by whichever contributor happened to be evaluated
        # first.
        return leaders[0], True, ClassificationDiagnostic.RESOLVED_BY_STABLE_ORDER
    return (
        leaders[0],
        distinct_candidates,
        ClassificationDiagnostic.CONTRIBUTORS_DISAGREE
        if distinct_candidates
        else ClassificationDiagnostic.RESOLVED_BY_STRENGTH,
    )
