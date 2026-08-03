"""Closed vocabularies for the domain layer.

Every enum here parses through :meth:`from_value`, which raises rather than
falling back to a default, so an unrecognized input fails closed.
"""

from __future__ import annotations

from enum import Enum
from typing import TypeVar

from .errors import UnsupportedValueError

__all__ = [
    "RoutingAction",
    "MessageCategory",
    "Provenance",
    "MediaKind",
    "ExtractionStatus",
    "Reliability",
    "ConversationSurface",
    "AbsenceReason",
    "AnalysisWarning",
    "RiskCategory",
    "TrustCategory",
    "parse_enum",
]

_E = TypeVar("_E", bound=Enum)


def parse_enum(enum_cls: type[_E], raw: object, field: str) -> _E:
    """Parse ``raw`` into a member of ``enum_cls``, or fail closed.

    Never guesses and never falls back to a default member.
    """
    for member in enum_cls:
        if member.value == raw:
            return member
    raise UnsupportedValueError(field)


class _ParsableEnum(Enum):
    """Mixin giving every vocabulary an identical fail-closed parser."""

    @classmethod
    def from_value(cls, raw: object, field: str | None = None):
        return parse_enum(cls, raw, field or cls.__name__)


class RoutingAction(_ParsableEnum):
    """How a message should reach its recipient.

    This is the routing outcome only. It says nothing about *what kind* of
    message this is -- that is :class:`MessageCategory`, an independent
    vocabulary. The two are never derived from one another.
    """

    #: Reach the recipient now.
    INTERRUPT = "interrupt"
    #: Hold for the recipient's next batched review.
    DEFER = "defer"
    #: Withhold from the recipient.
    SUPPRESS = "suppress"


class MessageCategory(_ParsableEnum):
    """A deliberately small starting vocabulary for what a message *is*.

    Kept coarse on purpose: it is a foundation meant to be widened later as
    real classification needs appear, not a finished taxonomy. Extending it
    is additive -- callers must handle unknown members defensively rather
    than assuming this list is exhaustive.
    """

    #: Back-and-forth between people.
    CONVERSATIONAL = "conversational"
    #: Tied to an account, order, booking, or payment.
    TRANSACTIONAL = "transactional"
    #: Sent to many recipients rather than to this one specifically.
    BROADCAST = "broadcast"
    #: Deliberately unassigned; never a silent fallback for a parse failure.
    UNCLASSIFIED = "unclassified"


class Provenance(_ParsableEnum):
    """Where a piece of content or signal came from.

    Provenance is a trust boundary, and it travels with the value it
    describes for the whole lifetime of that value. Nothing in this package
    lets a caller drop or overwrite it while keeping the content.
    """

    #: Fields supplied by the surrounding application (identifiers, flags).
    STRUCTURED_METADATA = "structured_metadata"
    #: Text a human typed into the message itself.
    AUTHORED_TEXT = "authored_text"
    #: Text read off an image by an OCR backend.
    OCR_TEXT = "ocr_text"
    #: Text produced from audio by a speech-recognition backend.
    ASR_TRANSCRIPT = "asr_transcript"
    #: Content obtained by resolving a link.
    URL_DERIVED = "url_derived"
    #: Content decoded from a scannable code.
    QR_DERIVED = "qr_derived"
    #: An interpretation produced by a model rather than observed directly.
    MODEL_INTERPRETATION = "model_interpretation"
    #: The output of a deterministic, reviewable check over other inputs.
    DERIVED_SIGNAL = "derived_signal"

    @property
    def is_extracted(self) -> bool:
        """True when the value was produced *from* other content.

        Extraction is lossy and attacker-influenced; these provenances are
        the ones a caller should treat with the most suspicion.
        """
        return self in _EXTRACTED

    @property
    def is_content(self) -> bool:
        """True for provenances that describe raw content.

        A deterministic signal is a conclusion, not content, so it is the
        one member for which this is False.
        """
        return self is not Provenance.DERIVED_SIGNAL

    @property
    def carries_instruction_authority(self) -> bool:
        """Always False, for every provenance without exception.

        Content is data. No provenance -- not authored text, not structured
        metadata, not a model's own interpretation -- makes a value into an
        instruction this system obeys. This property exists so that
        assumption is written down and testable rather than implied.
        """
        return False

    @property
    def confers_verified_trust(self) -> bool:
        """Always False, for every provenance without exception.

        In particular, :attr:`STRUCTURED_METADATA` arriving from the
        surrounding application is *structured*, which is not the same as
        *verified*. Trust is asserted separately and explicitly through a
        :class:`~message_router.domain.content.TrustSignal`.
        """
        return False


_EXTRACTED = frozenset(
    {
        Provenance.OCR_TEXT,
        Provenance.ASR_TRANSCRIPT,
        Provenance.URL_DERIVED,
        Provenance.QR_DERIVED,
        Provenance.MODEL_INTERPRETATION,
    }
)


class MediaKind(_ParsableEnum):
    """The broad type of an attachment."""

    IMAGE = "image"
    AUDIO = "audio"
    #: Present so an unrecognized attachment is representable rather than
    #: forced into a wrong bucket or silently dropped.
    OTHER = "other"


class ExtractionStatus(_ParsableEnum):
    """The outcome of asking a backend to read an attachment.

    These are kept distinct because conflating them hides real failures:
    "the backend was missing" and "the backend looked and found nothing"
    are very different facts about the same empty result.
    """

    NOT_ATTEMPTED = "not_attempted"
    SUCCEEDED = "succeeded"
    NO_CONTENT_FOUND = "no_content_found"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    FAILED = "failed"

    @property
    def yields_content(self) -> bool:
        """True only for the one status that may carry extracted content."""
        return self is ExtractionStatus.SUCCEEDED


class Reliability(_ParsableEnum):
    """How much weight a value's source can bear."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConversationSurface(_ParsableEnum):
    """The shape of the conversation a message arrived in."""

    DIRECT = "direct"
    GROUP = "group"
    CHANNEL = "channel"


class AbsenceReason(_ParsableEnum):
    """Why an optional piece of context is missing.

    Recording the reason keeps "we did not look" separate from "we looked
    and there was nothing", without inventing a placeholder value.
    """

    NOT_PROVIDED = "not_provided"
    NOT_APPLICABLE = "not_applicable"
    LOOKUP_FAILED = "lookup_failed"


class AnalysisWarning(_ParsableEnum):
    """Sanitized categories describing a degraded extraction."""

    LOW_CONFIDENCE_SOURCE = "low_confidence_source"
    NO_CONFIDENCE_REPORTED = "no_confidence_reported"
    PARTIAL_RESULT = "partial_result"
    UNSUPPORTED_FORMAT = "unsupported_format"
    TIMED_OUT = "timed_out"


class RiskCategory(_ParsableEnum):
    """Coarse risk families a structured signal can report.

    Detection itself is out of scope here; this is only the vocabulary a
    detector would report in.
    """

    IMPERSONATION = "impersonation"
    CREDENTIAL_SOLICITATION = "credential_solicitation"
    FINANCIAL_PRESSURE = "financial_pressure"
    UNSOLICITED_BULK = "unsolicited_bulk"
    EMBEDDED_DIRECTIVE = "embedded_directive"


class TrustCategory(_ParsableEnum):
    """Coarse families of positive, explicitly-asserted trust."""

    VERIFIED_ACCOUNT = "verified_account"
    ESTABLISHED_RELATIONSHIP = "established_relationship"
    RECIPIENT_INITIATED = "recipient_initiated"
