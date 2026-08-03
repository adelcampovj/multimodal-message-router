"""Controlled, sanitized error types for the domain layer.

Every error names *what kind* of rule was broken and *which field* broke it.
No error ever carries the offending value: message bodies, extracted text,
transcripts, decoded payloads, secrets, and filesystem paths must never
reach an exception string, its ``repr``, or a traceback rendering.

Formatting is deterministic so tests and logs can assert on it exactly.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "DomainError",
    "ValidationError",
    "ValidationIssue",
    "UnsupportedValueError",
]


class ValidationIssue(Enum):
    """The category of rule that was violated.

    Categories are deliberately coarse: they describe the *shape* of the
    failure, never the data that caused it.
    """

    EMPTY_IDENTIFIER = "empty_identifier"
    DUPLICATE_IDENTIFIER = "duplicate_identifier"
    EMPTY_TEXT = "empty_text"
    MISSING_CONTENT = "missing_content"
    OUT_OF_RANGE = "out_of_range"
    NOT_FINITE = "not_finite"
    NOT_A_NUMBER = "not_a_number"
    NAIVE_TIMESTAMP = "naive_timestamp"
    UNKNOWN_ENUM_VALUE = "unknown_enum_value"
    MEDIA_KIND_MISMATCH = "media_kind_mismatch"
    PROVENANCE_MISMATCH = "provenance_mismatch"
    RECIPIENT_SCOPE_MISMATCH = "recipient_scope_mismatch"
    CHRONOLOGY_VIOLATION = "chronology_violation"
    STATUS_CONTENT_MISMATCH = "status_content_mismatch"
    EMBEDDED_PATH = "embedded_path"
    WRONG_TYPE = "wrong_type"


class DomainError(Exception):
    """Base class for every error raised by this package's domain layer."""


class ValidationError(DomainError):
    """A domain invariant was violated.

    Carries only an issue category and a field name -- never a value.
    """

    def __init__(self, issue: ValidationIssue, field: str) -> None:
        self.issue = issue
        self.field = field
        super().__init__(self._render())

    def _render(self) -> str:
        return f"{self.issue.value}: field={self.field}"

    def __str__(self) -> str:  # deterministic, value-free
        return self._render()

    def __repr__(self) -> str:  # deterministic, value-free
        return f"ValidationError(issue={self.issue.value!r}, field={self.field!r})"


class UnsupportedValueError(ValidationError):
    """A closed vocabulary (enum) was given a value it does not define.

    Raised instead of silently falling back to a default, so unknown inputs
    fail closed rather than being absorbed.
    """

    def __init__(self, field: str) -> None:
        super().__init__(ValidationIssue.UNKNOWN_ENUM_VALUE, field)

    def __repr__(self) -> str:
        return f"UnsupportedValueError(field={self.field!r})"
