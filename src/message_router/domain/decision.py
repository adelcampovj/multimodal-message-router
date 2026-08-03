"""The outcome of routing one message.

:class:`RoutingDecision` binds together four things that are produced
independently and are meant to stay that way:

* the **action** -- how the message should reach the recipient;
* the **category** -- what kind of message it is;
* the **confidence** -- how firmly the action is held;
* the **rationale** -- why, in words a person can read.

Action and category are separate vocabularies and neither is derived from
the other here. A category is a description; an action is a decision; a bug
in one must not be able to silently rewrite the other.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..validation.invariants import require_text, require_unit_interval
from .content import RiskSignal, TrustSignal
from .enums import MessageCategory, RoutingAction
from .errors import ValidationError, ValidationIssue
from .evidence import EvidenceReference

__all__ = ["Confidence", "Rationale", "RoutingDecision"]


@dataclass(frozen=True)
class Confidence:
    """A finite score in ``[0.0, 1.0]``.

    Both endpoints are valid. NaN and the infinities are rejected outright
    rather than compared, since a silent non-finite value would make every
    downstream threshold test quietly false.
    """

    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", require_unit_interval(self.value, "value"))


@dataclass(frozen=True)
class Rationale:
    """A short human-readable explanation, stored trimmed and non-empty.

    This is written by the system about its own decision. It is not a place
    to quote message content, and callers should keep it that way.
    """

    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", require_text(self.text, "text"))


@dataclass(frozen=True)
class RoutingDecision:
    """A complete, self-describing routing outcome for one message."""

    message_id: str
    action: RoutingAction
    category: MessageCategory
    confidence: Confidence
    rationale: Rationale
    evidence: tuple[EvidenceReference, ...] = ()
    risk_signals: tuple[RiskSignal, ...] = ()
    trust_signals: tuple[TrustSignal, ...] = ()

    def __post_init__(self) -> None:
        from ..validation.invariants import require_identifier

        object.__setattr__(self, "message_id", require_identifier(self.message_id, "message_id"))
        if not isinstance(self.action, RoutingAction):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "action")
        if not isinstance(self.category, MessageCategory):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "category")
        if not isinstance(self.confidence, Confidence):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "confidence")
        if not isinstance(self.rationale, Rationale):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "rationale")

        for name, expected in (
            ("evidence", EvidenceReference),
            ("risk_signals", RiskSignal),
            ("trust_signals", TrustSignal),
        ):
            items = tuple(getattr(self, name))
            object.__setattr__(self, name, items)
            for item in items:
                if not isinstance(item, expected):
                    raise ValidationError(ValidationIssue.WRONG_TYPE, name)

    @property
    def cites_evidence(self) -> bool:
        """True only when at least one prior record was actually cited."""
        return bool(self.evidence)
