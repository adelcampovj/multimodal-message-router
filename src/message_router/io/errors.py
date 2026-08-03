"""Load-time failures, separated by the stage that produced them.

Knowing *where* loading broke is what makes a failure actionable: a
malformed byte stream, an unsupported version, a bad field, and a
contradiction between otherwise-valid fields call for four different fixes.

Every error names a location using a dotted pointer into the document
(``request.message.sent_at``). None of them ever carries a value from the
document -- not the offending field, not the message body, not a source
string. Pointers are safe to log; values are not.
"""

from __future__ import annotations

__all__ = [
    "LoadError",
    "DocumentParseError",
    "SchemaVersionError",
    "FieldError",
    "UnknownFieldError",
    "RelationshipError",
]


class LoadError(Exception):
    """Base class for every failure while loading a request document."""

    stage = "load"

    def __init__(self, pointer: str, detail: str) -> None:
        self.pointer = pointer
        self.detail = detail
        super().__init__(self._render())

    def _render(self) -> str:
        return f"{self.stage}/{self.detail}: at={self.pointer}"

    def __str__(self) -> str:
        return self._render()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(pointer={self.pointer!r}, detail={self.detail!r})"


class DocumentParseError(LoadError):
    """The bytes were not well-formed JSON.

    The underlying parser message is deliberately not reused: it quotes the
    surrounding document text, which may be message content.
    """

    stage = "parse"

    def __init__(self, pointer: str = "document") -> None:
        super().__init__(pointer, "malformed_json")


class SchemaVersionError(LoadError):
    """The document is not a version this build understands."""

    stage = "schema"

    def __init__(self, pointer: str, detail: str = "unsupported_version") -> None:
        super().__init__(pointer, detail)


class FieldError(LoadError):
    """A field was missing, the wrong type, or an unacceptable value."""

    stage = "field"


class UnknownFieldError(LoadError):
    """A key the format does not define was present.

    See :data:`~message_router.io.schema.UNKNOWN_FIELD_POLICY`. The
    unrecognised key name is reported in the pointer because it is a schema
    name the author wrote, not data.
    """

    stage = "field"

    def __init__(self, pointer: str) -> None:
        super().__init__(pointer, "unknown_field")


class RelationshipError(LoadError):
    """Individually valid fields contradict one another.

    Duplicate identities, or a message carrying neither text nor an
    attachment, are shape problems no single field can be blamed for.
    """

    stage = "relationship"
