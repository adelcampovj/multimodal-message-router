"""Content and signals, each permanently bound to its provenance.

The central rule of this package: **a string never travels without the
record of where it came from.** :class:`TextContent` is the only way to
carry text through the domain, and it cannot be constructed without a
:class:`~message_router.domain.enums.Provenance`.

Content is inert. Nothing in this module parses, evaluates, resolves, or
otherwise acts on the text it holds -- there is no branch anywhere that
reads a stored string and changes behavior because of what it says. Text
that happens to look like a directive, a flag assignment, or a trust claim
is stored, compared, and measured exactly like any other string.

Conclusions *about* content are modelled separately as
:class:`DeterministicSignal`, :class:`RiskSignal`, and :class:`TrustSignal`,
so a reviewable finding is never confused with the raw material it was
drawn from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..validation.invariants import require_identifier, require_text, require_unit_interval
from .enums import Provenance, Reliability, RiskCategory, TrustCategory

__all__ = [
    "TextContent",
    "DeterministicSignal",
    "RiskSignal",
    "TrustSignal",
]


@dataclass(frozen=True)
class TextContent:
    """An inert string plus the provenance it must always be read with.

    ``value`` is preserved byte-for-byte, including leading and trailing
    whitespace: this type never edits content, because normalizing here
    would quietly change what a later reviewer sees.

    The ``repr`` deliberately omits ``value`` and reports only its length,
    so content cannot leak into logs, test output, or tracebacks by
    accident. Read :attr:`value` explicitly when the content is genuinely
    needed.
    """

    value: str
    provenance: Provenance

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            from .errors import ValidationError, ValidationIssue

            raise ValidationError(ValidationIssue.WRONG_TYPE, "value")
        if not isinstance(self.provenance, Provenance):
            from .errors import ValidationError, ValidationIssue

            raise ValidationError(ValidationIssue.WRONG_TYPE, "provenance")

    @property
    def is_empty(self) -> bool:
        """True when the content holds nothing but whitespace."""
        return not self.value.strip()

    @property
    def is_extracted(self) -> bool:
        """True when this text was derived from other content."""
        return self.provenance.is_extracted

    @property
    def carries_instruction_authority(self) -> bool:
        """Always False -- content is data, whatever it happens to say."""
        return False

    def __repr__(self) -> str:
        return f"TextContent(provenance={self.provenance.value!r}, length={len(self.value)})"


@dataclass(frozen=True)
class DeterministicSignal:
    """A reviewable conclusion produced by a deterministic check.

    Distinct from :class:`TextContent` on purpose: this is an assertion the
    system is willing to stand behind and re-derive, not raw material. Its
    provenance is fixed to
    :attr:`~message_router.domain.enums.Provenance.DERIVED_SIGNAL` and
    cannot be set to a content provenance.

    ``detail`` is a short sanitized label describing the finding. It must
    never quote the content that triggered it.
    """

    name: str
    detail: str
    reliability: Reliability = Reliability.MEDIUM

    #: Fixed, not a constructor argument: a deterministic signal is never
    #: raw content, so its provenance is not the caller's to choose.
    provenance: Provenance = field(default=Provenance.DERIVED_SIGNAL, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_identifier(self.name, "name"))
        object.__setattr__(self, "detail", require_text(self.detail, "detail"))
        if not isinstance(self.reliability, Reliability):
            from .errors import ValidationError, ValidationIssue

            raise ValidationError(ValidationIssue.WRONG_TYPE, "reliability")


@dataclass(frozen=True)
class RiskSignal:
    """A structured, sanitized statement that something looks risky.

    Carries the provenance of the material the concern was raised from, so
    a reader can weigh a concern raised from a transcript differently from
    one raised from structured metadata.

    Detection logic lives outside this package; this is only how such a
    finding is represented.
    """

    category: RiskCategory
    detail: str
    observed_in: Provenance
    reliability: Reliability = Reliability.MEDIUM

    def __post_init__(self) -> None:
        from .errors import ValidationError, ValidationIssue

        if not isinstance(self.category, RiskCategory):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "category")
        if not isinstance(self.observed_in, Provenance):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "observed_in")
        if not isinstance(self.reliability, Reliability):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "reliability")
        object.__setattr__(self, "detail", require_text(self.detail, "detail"))


@dataclass(frozen=True)
class TrustSignal:
    """A structured, explicitly-asserted reason to extend trust.

    Trust is only ever positive evidence someone chose to record. It is
    never inferred from the fact that a value arrived as structured
    metadata -- see
    :attr:`Provenance.confers_verified_trust
    <message_router.domain.enums.Provenance.confers_verified_trust>`.
    """

    category: TrustCategory
    detail: str
    asserted_by: Provenance = Provenance.STRUCTURED_METADATA
    strength: float = 1.0

    def __post_init__(self) -> None:
        from .errors import ValidationError, ValidationIssue

        if not isinstance(self.category, TrustCategory):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "category")
        if not isinstance(self.asserted_by, Provenance):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "asserted_by")
        object.__setattr__(self, "detail", require_text(self.detail, "detail"))
        object.__setattr__(self, "strength", require_unit_interval(self.strength, "strength"))
