"""References to prior records offered as background for a decision.

A reference points at history; it does not carry history. There is no field
here for the text of a past message, and there is deliberately no scoring,
ranking, or relevance field -- selecting *which* references matter is a
separate concern handled later.

Two rules are enforced at construction and at use:

* a reference belongs to exactly one recipient, and cannot be attached to a
  decision made for anyone else;
* a reference must be shown to precede the moment it is cited for, rather
  than assumed to.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from ..validation.invariants import (
    normalize_timestamp,
    require_identifier,
    require_strictly_prior,
)
from .context import RecipientScope
from .errors import ValidationError, ValidationIssue

__all__ = ["EvidenceReference"]


@dataclass(frozen=True)
class EvidenceReference:
    """A pointer to one prior record belonging to one recipient."""

    record_id: str
    recipient: RecipientScope
    occurred_at: dt.datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", require_identifier(self.record_id, "record_id"))
        if not isinstance(self.recipient, RecipientScope):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "recipient")
        object.__setattr__(self, "occurred_at", normalize_timestamp(self.occurred_at, "occurred_at"))

    def belongs_to(self, recipient: RecipientScope) -> bool:
        """True when this reference is in ``recipient``'s scope."""
        return self.recipient.matches(recipient)

    def is_strictly_prior_to(self, moment: dt.datetime) -> bool:
        """True when this record precedes ``moment``.

        Non-raising counterpart to :meth:`require_usable_for`. A record
        stamped at exactly ``moment`` returns False -- simultaneity is not
        precedence.
        """
        return self.occurred_at < normalize_timestamp(moment, "moment")

    def require_usable_for(self, recipient: RecipientScope, moment: dt.datetime) -> None:
        """Raise unless this reference may be cited for ``recipient`` at ``moment``.

        Checks recipient scope first: citing another recipient's history is
        the more serious error, and reporting it as a chronology problem
        would be misleading.
        """
        if not self.belongs_to(recipient):
            raise ValidationError(ValidationIssue.RECIPIENT_SCOPE_MISMATCH, "recipient")
        require_strictly_prior(self.occurred_at, moment, "occurred_at")
