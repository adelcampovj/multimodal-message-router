"""Reusable invariant checks shared by the domain objects.

These are *structural* guarantees only -- identifiers are present, numbers
are in range, timestamps are unambiguous, references line up with what they
claim to reference. They keep malformed input from silently becoming a
valid-looking object.

They are deliberately **not** a safety control. Nothing here inspects
content for hostile intent, and passing every check in this module implies
nothing about whether a message is safe.
"""

from __future__ import annotations

import datetime as dt
import math

from ..domain.errors import ValidationError, ValidationIssue

__all__ = [
    "require_identifier",
    "require_text",
    "require_unit_interval",
    "require_aware_timestamp",
    "normalize_timestamp",
    "require_no_embedded_path",
    "require_strictly_prior",
]

#: Timestamp normalization policy for the whole package.
#:
#: Every timestamp entering the domain must already carry a UTC offset; a
#: naive timestamp is rejected rather than assigned a guessed zone. Accepted
#: timestamps are then converted to UTC so that ordering comparisons between
#: values from different sources are exact and total.
TIMESTAMP_POLICY = "aware-required; normalized to UTC"

_PATH_MARKERS = ("/", "\\", "://")


def require_identifier(value: object, field: str) -> str:
    """Return ``value`` trimmed, rejecting non-strings and blank strings.

    Identifiers are the join keys of this domain; a blank or whitespace-only
    one would quietly link unrelated records together.
    """
    if not isinstance(value, str):
        raise ValidationError(ValidationIssue.WRONG_TYPE, field)
    trimmed = value.strip()
    if not trimmed:
        raise ValidationError(ValidationIssue.EMPTY_IDENTIFIER, field)
    return trimmed


def require_text(value: object, field: str) -> str:
    """Return ``value`` trimmed, rejecting non-strings and blank strings."""
    if not isinstance(value, str):
        raise ValidationError(ValidationIssue.WRONG_TYPE, field)
    trimmed = value.strip()
    if not trimmed:
        raise ValidationError(ValidationIssue.EMPTY_TEXT, field)
    return trimmed


def require_unit_interval(value: object, field: str) -> float:
    """Return ``value`` as a float in ``[0.0, 1.0]``.

    Rejects NaN and both infinities before the range test, so a non-finite
    value can never slip through a comparison that silently returns False.
    Booleans are rejected: ``True`` is not a confidence of 1.0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(ValidationIssue.WRONG_TYPE, field)
    numeric = float(value)
    if math.isnan(numeric):
        raise ValidationError(ValidationIssue.NOT_A_NUMBER, field)
    if math.isinf(numeric):
        raise ValidationError(ValidationIssue.NOT_FINITE, field)
    if not 0.0 <= numeric <= 1.0:
        raise ValidationError(ValidationIssue.OUT_OF_RANGE, field)
    return numeric


def require_aware_timestamp(value: object, field: str) -> dt.datetime:
    """Return ``value`` unchanged after proving it carries a UTC offset.

    A naive timestamp is ambiguous, so it is rejected rather than given an
    assumed zone. Use :func:`normalize_timestamp` to also convert to UTC.
    """
    if not isinstance(value, dt.datetime):
        raise ValidationError(ValidationIssue.WRONG_TYPE, field)
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValidationError(ValidationIssue.NAIVE_TIMESTAMP, field)
    return value


def normalize_timestamp(value: object, field: str) -> dt.datetime:
    """Validate ``value`` is aware, then convert it to UTC.

    See :data:`TIMESTAMP_POLICY`. Normalizing on the way in means every
    stored timestamp is directly comparable with every other one.
    """
    return require_aware_timestamp(value, field).astimezone(dt.timezone.utc)


def require_no_embedded_path(value: object, field: str) -> str:
    """Return a trimmed string, rejecting anything that looks like a path.

    Used for backend identities, which should name an engine, not disclose
    where it happens to live on the machine that ran it.
    """
    trimmed = require_identifier(value, field)
    lowered = trimmed.lower()
    if any(marker in lowered for marker in _PATH_MARKERS):
        raise ValidationError(ValidationIssue.EMBEDDED_PATH, field)
    if len(trimmed) >= 2 and trimmed[1] == ":" and trimmed[0].isalpha():
        raise ValidationError(ValidationIssue.EMBEDDED_PATH, field)
    return trimmed


def require_strictly_prior(earlier: dt.datetime, reference: dt.datetime, field: str) -> None:
    """Raise unless ``earlier`` is strictly before ``reference``.

    Equal timestamps are rejected: a record stamped at exactly the moment of
    the message it is offered as background for cannot be established as
    having preceded it.
    """
    left = normalize_timestamp(earlier, field)
    right = normalize_timestamp(reference, field)
    if left >= right:
        raise ValidationError(ValidationIssue.CHRONOLOGY_VIOLATION, field)
