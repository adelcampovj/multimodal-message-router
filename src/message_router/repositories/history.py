"""Recipient-scoped access to prior records.

The one query this repository offers requires both a recipient and a
cutoff. There is deliberately no "fetch everything" method, no
cross-recipient lookup, and no way to ask for records without saying which
recipient they are for -- a convenience accessor is exactly how one
recipient's history ends up in another's decision.

Filtering only. Choosing which of the returned references actually matter
is a separate concern and is not implemented here: there is no scoring, no
ranking, no similarity, and no message text to compare in the first place.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence

from ..domain.context import RecipientScope
from ..domain.errors import ValidationError, ValidationIssue
from ..domain.evidence import EvidenceReference
from ..validation.invariants import normalize_timestamp

__all__ = ["InMemoryHistoryRepository"]


class InMemoryHistoryRepository:
    """Holds :class:`EvidenceReference` records and serves scoped queries.

    Satisfies the
    :class:`~message_router.ports.backends.EvidenceRepository` Protocol
    structurally, without importing or subclassing it.
    """

    def __init__(self, references: Iterable[EvidenceReference] = ()) -> None:
        seen: set[str] = set()
        stored: list[EvidenceReference] = []
        for reference in references:
            if not isinstance(reference, EvidenceReference):
                raise ValidationError(ValidationIssue.WRONG_TYPE, "references")
            if reference.record_id in seen:
                # Two records sharing an identity make any later citation
                # ambiguous, so this is refused at construction rather than
                # silently de-duplicated to whichever arrived first.
                raise ValidationError(ValidationIssue.DUPLICATE_IDENTIFIER, "record_id")
            seen.add(reference.record_id)
            stored.append(reference)
        self._references = tuple(stored)

    def __len__(self) -> int:
        return len(self._references)

    def references_before(
        self, recipient: RecipientScope, moment: dt.datetime, limit: int
    ) -> Sequence[EvidenceReference]:
        """Return ``recipient``'s records from strictly before ``moment``.

        Ordering is newest first, then by record identity, so the result is
        fully determined by the stored data and never by insertion order or
        dictionary iteration.

        A record stamped exactly at ``moment`` is excluded: simultaneity is
        not precedence.
        """
        if not isinstance(recipient, RecipientScope):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "recipient")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "limit")
        if limit < 0:
            raise ValidationError(ValidationIssue.OUT_OF_RANGE, "limit")
        cutoff = normalize_timestamp(moment, "moment")

        matching = [
            reference
            for reference in self._references
            if reference.belongs_to(recipient) and reference.occurred_at < cutoff
        ]
        matching.sort(key=lambda reference: (-reference.occurred_at.timestamp(), reference.record_id))
        return tuple(matching[:limit])
