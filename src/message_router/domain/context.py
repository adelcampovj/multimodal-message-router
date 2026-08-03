"""Who a message is for, who sent it, and what surrounds it.

Every element here is optional-by-omission rather than optional-by-
placeholder: when a sender or a preference set is unknown, the field is
``None`` and the gap may be recorded explicitly as a :class:`ContextGap`.
Nothing in this module invents a stand-in value to fill a hole.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from ..validation.invariants import (
    normalize_timestamp,
    require_identifier,
    require_text,
)
from .content import TrustSignal
from .enums import AbsenceReason, ConversationSurface

__all__ = [
    "RecipientScope",
    "SenderContext",
    "ConversationContext",
    "QuietPeriod",
    "RecipientPreferences",
    "InteractionRecord",
    "ContextGap",
]


@dataclass(frozen=True)
class RecipientScope:
    """The single recipient a decision is being made for.

    This is the boundary that background records may never cross: anything
    offered as history for this recipient must carry an equal scope. Two
    scopes are equal exactly when their identifiers are equal.
    """

    recipient_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "recipient_id", require_identifier(self.recipient_id, "recipient_id"))

    def matches(self, other: "RecipientScope") -> bool:
        """True when ``other`` refers to the same recipient."""
        return isinstance(other, RecipientScope) and other.recipient_id == self.recipient_id


@dataclass(frozen=True)
class SenderContext:
    """What is known about the party that sent a message.

    Deliberately holds no display name. A human-facing label is attacker-
    controlled and invites treating a familiar-looking name as evidence of
    identity; trust belongs in :attr:`trust_signals`, which someone had to
    assert on purpose.
    """

    sender_id: str
    trust_signals: tuple[TrustSignal, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sender_id", require_identifier(self.sender_id, "sender_id"))
        object.__setattr__(self, "trust_signals", tuple(self.trust_signals))

    @property
    def has_asserted_trust(self) -> bool:
        """True only when at least one trust signal was explicitly recorded."""
        return bool(self.trust_signals)


@dataclass(frozen=True)
class ConversationContext:
    """The conversation a message arrived in."""

    conversation_id: str
    surface: ConversationSurface
    participant_count: int | None = None

    def __post_init__(self) -> None:
        from .errors import ValidationError, ValidationIssue

        object.__setattr__(
            self, "conversation_id", require_identifier(self.conversation_id, "conversation_id")
        )
        if not isinstance(self.surface, ConversationSurface):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "surface")
        if self.participant_count is not None:
            if isinstance(self.participant_count, bool) or not isinstance(self.participant_count, int):
                raise ValidationError(ValidationIssue.WRONG_TYPE, "participant_count")
            if self.participant_count < 0:
                raise ValidationError(ValidationIssue.OUT_OF_RANGE, "participant_count")


@dataclass(frozen=True)
class QuietPeriod:
    """A recurring local-time window during which a recipient prefers quiet.

    Stored as whole minutes past midnight. The window may wrap past
    midnight, which is why containment is not a simple ``<=`` chain.
    """

    start_minute: int
    end_minute: int

    def __post_init__(self) -> None:
        from .errors import ValidationError, ValidationIssue

        for name in ("start_minute", "end_minute"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ValidationError(ValidationIssue.WRONG_TYPE, name)
            if not 0 <= raw < 24 * 60:
                raise ValidationError(ValidationIssue.OUT_OF_RANGE, name)

    @property
    def wraps_midnight(self) -> bool:
        return self.start_minute > self.end_minute

    def contains_minute(self, minute_of_day: int) -> bool:
        """True when ``minute_of_day`` falls inside the window."""
        if self.start_minute == self.end_minute:
            return False
        if self.wraps_midnight:
            return minute_of_day >= self.start_minute or minute_of_day < self.end_minute
        return self.start_minute <= minute_of_day < self.end_minute


@dataclass(frozen=True)
class RecipientPreferences:
    """Choices a recipient made about how they want to be reached."""

    quiet_period: QuietPeriod | None = None
    muted_conversation_ids: frozenset[str] = frozenset()
    accepts_broadcast: bool = True

    def __post_init__(self) -> None:
        from .errors import ValidationError, ValidationIssue

        if self.quiet_period is not None and not isinstance(self.quiet_period, QuietPeriod):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "quiet_period")
        if not isinstance(self.accepts_broadcast, bool):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "accepts_broadcast")
        cleaned = frozenset(
            require_identifier(value, "muted_conversation_ids") for value in self.muted_conversation_ids
        )
        object.__setattr__(self, "muted_conversation_ids", cleaned)

    def has_muted(self, conversation_id: str) -> bool:
        return conversation_id in self.muted_conversation_ids


@dataclass(frozen=True)
class InteractionRecord:
    """One past interaction between a recipient and a conversation.

    Holds counts and outcomes only. It intentionally carries no message
    text: behavioural history is useful for routing without reproducing
    what anyone actually wrote.
    """

    recipient: RecipientScope
    conversation_id: str
    occurred_at: dt.datetime
    opened: bool = False
    replied: bool = False
    dismissed: bool = False

    def __post_init__(self) -> None:
        from .errors import ValidationError, ValidationIssue

        if not isinstance(self.recipient, RecipientScope):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "recipient")
        object.__setattr__(
            self, "conversation_id", require_identifier(self.conversation_id, "conversation_id")
        )
        object.__setattr__(self, "occurred_at", normalize_timestamp(self.occurred_at, "occurred_at"))
        for name in ("opened", "replied", "dismissed"):
            if not isinstance(getattr(self, name), bool):
                raise ValidationError(ValidationIssue.WRONG_TYPE, name)


@dataclass(frozen=True)
class ContextGap:
    """A recorded, deliberate absence.

    Lets a caller distinguish "nobody looked" from "we looked and there was
    nothing" without either case being represented by an invented value.
    """

    element: str
    reason: AbsenceReason

    def __post_init__(self) -> None:
        from .errors import ValidationError, ValidationIssue

        object.__setattr__(self, "element", require_text(self.element, "element"))
        if not isinstance(self.reason, AbsenceReason):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "reason")
