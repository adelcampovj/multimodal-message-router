"""Turning a request document into validated domain objects.

Parsing here means :func:`json.loads` and nothing else: no evaluation, no
imports driven by content, no dynamic attribute lookup from document keys.
Values from the document are only ever passed as data into the domain
constructors, which then apply their own invariants -- so a document cannot
produce an object the domain would have rejected.

Content is inert throughout. A body, a source string, or a history entry
that reads like an instruction is stored, validated for shape, and
otherwise ignored. Nothing branches on what a string says.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..domain.content import TextContent
from ..domain.context import (
    ConversationContext,
    QuietPeriod,
    RecipientPreferences,
    RecipientScope,
    SenderContext,
)
from ..domain.enums import ConversationSurface, MediaKind, Provenance
from ..domain.errors import ValidationError
from ..domain.evidence import EvidenceReference
from ..domain.media import MediaReference
from ..domain.message import IncomingMessage
from .errors import (
    DocumentParseError,
    FieldError,
    RelationshipError,
    SchemaVersionError,
    UnknownFieldError,
)
from .schema import (
    ATTACHMENT_FIELDS,
    CONVERSATION_FIELDS,
    DOCUMENT_FIELDS,
    FORMAT_NAME,
    HISTORY_FIELDS,
    MESSAGE_FIELDS,
    PREFERENCE_FIELDS,
    RECIPIENT_FIELDS,
    REQUEST_FIELDS,
    SENDER_FIELDS,
    SUPPORTED_VERSIONS,
)

__all__ = ["LoadedRequest", "load_request", "load_request_file"]


@dataclass(frozen=True)
class LoadedRequest:
    """Everything one document described, as validated domain objects.

    ``media_sources`` is kept beside the message rather than inside it:
    :class:`~message_router.domain.media.MediaReference` intentionally
    carries no location, so the relative source string lives here, where a
    caller must pair it with an explicit media root to resolve it.
    """

    message: IncomingMessage
    preferences: RecipientPreferences | None = None
    history: tuple[EvidenceReference, ...] = ()
    media_sources: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Small typed readers. Each reports its own pointer, never its value.
# ---------------------------------------------------------------------------


def _require_mapping(value: object, pointer: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise FieldError(pointer, "expected_object")
    return value


def _reject_unknown(value: Mapping, allowed: frozenset[str], pointer: str) -> None:
    for key in value:
        if key not in allowed:
            raise UnknownFieldError(f"{pointer}.{key}")


def _require(value: Mapping, key: str, pointer: str) -> object:
    if key not in value:
        raise FieldError(f"{pointer}.{key}", "missing")
    return value[key]


def _read_str(value: Mapping, key: str, pointer: str) -> str:
    raw = _require(value, key, pointer)
    if not isinstance(raw, str):
        raise FieldError(f"{pointer}.{key}", "expected_string")
    return raw


def _read_optional_str(value: Mapping, key: str, pointer: str) -> str | None:
    if key not in value or value[key] is None:
        return None
    raw = value[key]
    if not isinstance(raw, str):
        raise FieldError(f"{pointer}.{key}", "expected_string")
    return raw


def _read_bool(value: Mapping, key: str, pointer: str, default: bool) -> bool:
    if key not in value or value[key] is None:
        return default
    raw = value[key]
    if not isinstance(raw, bool):
        raise FieldError(f"{pointer}.{key}", "expected_boolean")
    return raw


def _read_int(value: Mapping, key: str, pointer: str) -> int | None:
    if key not in value or value[key] is None:
        return None
    raw = value[key]
    # `True` is an int in Python; accepting it here would silently turn a
    # boolean into the number 1.
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise FieldError(f"{pointer}.{key}", "expected_integer")
    return raw


def _read_timestamp(value: Mapping, key: str, pointer: str) -> dt.datetime:
    raw = _read_str(value, key, pointer)
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        raise FieldError(f"{pointer}.{key}", "expected_iso_timestamp") from None
    if parsed.tzinfo is None:
        # The domain's timestamp policy requires an offset; guessing one
        # here would be inventing information the document did not supply.
        raise FieldError(f"{pointer}.{key}", "expected_offset_aware_timestamp")
    return parsed


def _read_enum(enum_cls, value: Mapping, key: str, pointer: str):
    raw = _read_str(value, key, pointer)
    for member in enum_cls:
        if member.value == raw:
            return member
    raise FieldError(f"{pointer}.{key}", "unsupported_value")


def _wrap_domain(pointer: str, build):
    """Run a domain constructor, restating its failure as a load failure.

    The domain error's issue category is carried across; its field name is
    replaced by the document pointer so the message points at the document
    the author can actually edit.
    """
    try:
        return build()
    except ValidationError as exc:
        raise FieldError(pointer, exc.issue.value) from None


# ---------------------------------------------------------------------------
# Section readers.
# ---------------------------------------------------------------------------


def _load_preferences(raw: object, pointer: str) -> RecipientPreferences | None:
    if raw is None:
        return None
    value = _require_mapping(raw, pointer)
    _reject_unknown(value, PREFERENCE_FIELDS, pointer)

    start = _read_int(value, "quiet_from_minute", pointer)
    end = _read_int(value, "quiet_to_minute", pointer)
    if (start is None) != (end is None):
        raise RelationshipError(pointer, "quiet_period_requires_both_bounds")
    quiet = None
    if start is not None and end is not None:
        quiet = _wrap_domain(f"{pointer}.quiet_from_minute", lambda: QuietPeriod(start, end))

    silenced = value.get("silenced_conversations") or []
    if not isinstance(silenced, list):
        raise FieldError(f"{pointer}.silenced_conversations", "expected_array")
    for entry in silenced:
        if not isinstance(entry, str):
            raise FieldError(f"{pointer}.silenced_conversations", "expected_string")

    accepts = _read_bool(value, "accepts_broadcast", pointer, True)
    return _wrap_domain(
        pointer,
        lambda: RecipientPreferences(
            quiet_period=quiet,
            muted_conversation_ids=frozenset(silenced),
            accepts_broadcast=accepts,
        ),
    )


def _load_recipient(raw: object, pointer: str) -> tuple[RecipientScope, RecipientPreferences | None]:
    value = _require_mapping(raw, pointer)
    _reject_unknown(value, RECIPIENT_FIELDS, pointer)
    scope = _wrap_domain(
        f"{pointer}.id", lambda: RecipientScope(_read_str(value, "id", pointer))
    )
    preferences = _load_preferences(value.get("preferences"), f"{pointer}.preferences")
    return scope, preferences


def _load_conversation(raw: object, pointer: str) -> ConversationContext:
    value = _require_mapping(raw, pointer)
    _reject_unknown(value, CONVERSATION_FIELDS, pointer)
    identifier = _read_str(value, "id", pointer)
    surface = _read_enum(ConversationSurface, value, "surface", pointer)
    participants = _read_int(value, "participants", pointer)
    return _wrap_domain(
        pointer, lambda: ConversationContext(identifier, surface, participants)
    )


def _load_sender(raw: object, pointer: str) -> SenderContext | None:
    if raw is None:
        return None
    value = _require_mapping(raw, pointer)
    _reject_unknown(value, SENDER_FIELDS, pointer)
    identifier = _read_str(value, "id", pointer)
    return _wrap_domain(pointer, lambda: SenderContext(identifier))


def _load_attachment(raw: object, pointer: str) -> tuple[MediaReference, str] | None:
    if raw is None:
        return None
    value = _require_mapping(raw, pointer)
    _reject_unknown(value, ATTACHMENT_FIELDS, pointer)
    identifier = _read_str(value, "id", pointer)
    kind = _read_enum(MediaKind, value, "kind", pointer)
    source = _read_str(value, "source", pointer)
    reference = _wrap_domain(pointer, lambda: MediaReference(identifier, kind))
    return reference, source


def _load_message(raw: object, pointer: str, recipient: RecipientScope,
                  conversation: ConversationContext, sender: SenderContext | None):
    value = _require_mapping(raw, pointer)
    _reject_unknown(value, MESSAGE_FIELDS, pointer)

    identifier = _read_str(value, "id", pointer)
    sent_at = _read_timestamp(value, "sent_at", pointer)

    raw_body = _read_optional_str(value, "body", pointer)
    body = None
    if raw_body is not None:
        # An empty or whitespace-only body is refused rather than stored:
        # it would be indistinguishable from a media-only message while
        # falsely asserting that someone wrote something.
        if not raw_body.strip():
            raise FieldError(f"{pointer}.body", "empty_text")
        body = TextContent(raw_body, Provenance.AUTHORED_TEXT)

    attachment_pair = _load_attachment(value.get("attachment"), f"{pointer}.attachment")
    attachment = attachment_pair[0] if attachment_pair else None

    if body is None and attachment is None:
        raise RelationshipError(pointer, "requires_body_or_attachment")

    message = _wrap_domain(
        pointer,
        lambda: IncomingMessage(
            message_id=identifier,
            recipient=recipient,
            conversation=conversation,
            sent_at=sent_at,
            body=body,
            sender=sender,
            attachment=attachment,
        ),
    )
    sources = {attachment_pair[0].media_id: attachment_pair[1]} if attachment_pair else {}
    return message, sources


def _load_history(raw: object, pointer: str, recipient: RecipientScope) -> tuple[EvidenceReference, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise FieldError(pointer, "expected_array")

    references: list[EvidenceReference] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        entry_pointer = f"{pointer}[{index}]"
        value = _require_mapping(entry, entry_pointer)
        _reject_unknown(value, HISTORY_FIELDS, entry_pointer)

        identifier = _read_str(value, "id", entry_pointer)
        occurred_at = _read_timestamp(value, "occurred_at", entry_pointer)
        # Present in the format and validated, but not carried into the
        # reference: an EvidenceReference is a pointer to a record, not a
        # copy of it.
        _read_optional_str(value, "conversation_id", entry_pointer)
        for flag in ("opened", "replied", "dismissed"):
            _read_bool(value, flag, entry_pointer, False)

        if identifier.strip() in seen:
            raise RelationshipError(entry_pointer, "duplicate_record_id")
        seen.add(identifier.strip())

        references.append(
            _wrap_domain(
                entry_pointer,
                lambda i=identifier, t=occurred_at: EvidenceReference(i, recipient, t),
            )
        )
    return tuple(references)


# ---------------------------------------------------------------------------
# Entry points.
# ---------------------------------------------------------------------------


def load_request(document: str | bytes) -> LoadedRequest:
    """Parse and validate one request document."""
    try:
        parsed = json.loads(document)
    except (ValueError, TypeError):
        raise DocumentParseError() from None

    value = _require_mapping(parsed, "document")
    _reject_unknown(value, DOCUMENT_FIELDS, "document")

    if _read_str(value, "format", "document") != FORMAT_NAME:
        raise SchemaVersionError("document.format", "unsupported_format")
    version = _read_int(value, "version", "document")
    if version is None:
        raise FieldError("document.version", "missing")
    if version not in SUPPORTED_VERSIONS:
        raise SchemaVersionError("document.version")

    request = _require_mapping(_require(value, "request", "document"), "request")
    _reject_unknown(request, REQUEST_FIELDS, "request")

    recipient, preferences = _load_recipient(_require(request, "recipient", "request"), "request.recipient")
    conversation = _load_conversation(_require(request, "conversation", "request"), "request.conversation")
    sender = _load_sender(request.get("sender"), "request.sender")
    message, media_sources = _load_message(
        _require(request, "message", "request"), "request.message", recipient, conversation, sender
    )
    history = _load_history(request.get("history"), "request.history", recipient)

    return LoadedRequest(
        message=message,
        preferences=preferences,
        history=history,
        media_sources=media_sources,
    )


def load_request_file(path: Path) -> LoadedRequest:
    """Read and load one request document from disk.

    An unreadable file raises a parse-stage failure naming the stage only;
    the path is not included.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError:
        raise DocumentParseError("document.file") from None
    return load_request(raw)
