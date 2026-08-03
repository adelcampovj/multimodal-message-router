"""The incoming message a routing decision is made about."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from ..validation.invariants import normalize_timestamp, require_identifier
from .content import TextContent
from .context import ConversationContext, ContextGap, RecipientScope, SenderContext
from .enums import Provenance
from .errors import ValidationError, ValidationIssue
from .media import MediaReference

__all__ = ["IncomingMessage"]


@dataclass(frozen=True)
class IncomingMessage:
    """One message addressed to one recipient.

    ``body`` is :class:`~message_router.domain.content.TextContent` rather
    than a bare string, so the message text carries its provenance
    everywhere it goes and is never mistaken for a trusted field.

    A message may be text-only, media-only, or both. ``body`` is therefore
    optional: a picture or a voice note sent with no caption has **no**
    authored text, and inventing an empty string for it would be a claim
    that someone wrote nothing when in fact they wrote nothing *down*.
    What is required is that at least one of ``body`` or ``attachment`` is
    present -- a message carrying neither is not a message.

    ``sender``, ``attachment``, and the rest stay ``None`` when unknown.
    Anything genuinely missing may be recorded in :attr:`gaps`, which
    explains an absence without filling it in.
    """

    message_id: str
    recipient: RecipientScope
    conversation: ConversationContext
    sent_at: dt.datetime
    body: TextContent | None = None
    sender: SenderContext | None = None
    attachment: MediaReference | None = None
    gaps: tuple[ContextGap, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "message_id", require_identifier(self.message_id, "message_id"))
        if not isinstance(self.recipient, RecipientScope):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "recipient")
        if not isinstance(self.conversation, ConversationContext):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "conversation")
        if self.body is not None:
            if not isinstance(self.body, TextContent):
                raise ValidationError(ValidationIssue.WRONG_TYPE, "body")
            if self.body.provenance is not Provenance.AUTHORED_TEXT:
                # The message body is what a human typed. Extracted or model-
                # generated text belongs in a MediaAnalysis or a signal, never
                # substituted in here as though a person had written it.
                raise ValidationError(ValidationIssue.PROVENANCE_MISMATCH, "body")
        object.__setattr__(self, "sent_at", normalize_timestamp(self.sent_at, "sent_at"))
        if self.sender is not None and not isinstance(self.sender, SenderContext):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "sender")
        if self.attachment is not None and not isinstance(self.attachment, MediaReference):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "attachment")
        object.__setattr__(self, "gaps", tuple(self.gaps))
        for gap in self.gaps:
            if not isinstance(gap, ContextGap):
                raise ValidationError(ValidationIssue.WRONG_TYPE, "gaps")

        if self.body is None and self.attachment is None:
            raise ValidationError(ValidationIssue.MISSING_CONTENT, "body")

    @property
    def has_known_sender(self) -> bool:
        """False when no sender context was supplied -- never a guess."""
        return self.sender is not None

    @property
    def has_attachment(self) -> bool:
        return self.attachment is not None

    @property
    def has_authored_text(self) -> bool:
        """False for a media-only message -- never softened to an empty string."""
        return self.body is not None
