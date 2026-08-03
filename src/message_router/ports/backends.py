"""Interfaces for the replaceable parts of the system.

These are :class:`~typing.Protocol` definitions, so an implementation
satisfies one by having the right methods -- it never imports from or
subclasses anything here. That keeps engine-specific packages out of this
one entirely: importing :mod:`message_router` requires no OCR engine, no
speech recogniser, no model client, no credentials, and no network.

Every port is written so failure is a *returned value*, not an exception.
An unavailable backend reports
:attr:`~message_router.domain.enums.ExtractionStatus.BACKEND_UNAVAILABLE`
and a failed one reports
:attr:`~message_router.domain.enums.ExtractionStatus.FAILED`, which means a
caller cannot accidentally treat a missing engine as "found nothing".

Because they are structural, test doubles are ordinary small classes -- no
base class, no registration, no patching.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol, Sequence, runtime_checkable

from ..domain.content import TextContent
from ..domain.context import RecipientScope
from ..domain.evidence import EvidenceReference
from ..domain.media import MediaAnalysis, MediaReference

__all__ = [
    "OcrBackend",
    "AsrBackend",
    "QrBackend",
    "EvidenceRepository",
    "SemanticAnalysisBackend",
    "Clock",
]


@runtime_checkable
class OcrBackend(Protocol):
    """Reads text out of an image attachment."""

    def extract_text(self, media: MediaReference) -> MediaAnalysis:
        """Return what was read, or a status explaining why nothing was.

        Implementations must not raise for an unreadable or unsupported
        attachment; they report it in the returned analysis instead.
        """
        ...


@runtime_checkable
class AsrBackend(Protocol):
    """Transcribes an audio attachment."""

    def transcribe(self, media: MediaReference) -> MediaAnalysis:
        """Return the transcript, or a status explaining its absence."""
        ...


@runtime_checkable
class QrBackend(Protocol):
    """Decodes scannable codes embedded in an image attachment."""

    def decode(self, media: MediaReference) -> MediaAnalysis:
        """Return any decoded payload, or a status explaining its absence.

        A decoded payload is content, not a command. Implementations must
        return it as
        :attr:`~message_router.domain.enums.Provenance.QR_DERIVED` content
        and must never resolve, open, fetch, or act on what it contains --
        a payload that looks like a link or an instruction is still just a
        string.
        """
        ...


@runtime_checkable
class EvidenceRepository(Protocol):
    """Supplies prior records belonging to one recipient."""

    def references_before(
        self, recipient: RecipientScope, moment: dt.datetime, limit: int
    ) -> Sequence[EvidenceReference]:
        """Return references for ``recipient`` from strictly before ``moment``.

        The recipient is a required argument rather than a filter applied
        afterwards, so an implementation cannot return another recipient's
        history by omission. Callers should still verify each reference
        with
        :meth:`EvidenceReference.require_usable_for
        <message_router.domain.evidence.EvidenceReference.require_usable_for>`
        rather than trusting the repository to have honoured the contract.
        """
        ...


@runtime_checkable
class SemanticAnalysisBackend(Protocol):
    """Optional model-based reading of content.

    Whatever this returns is an interpretation, not an observation, and is
    tagged
    :attr:`~message_router.domain.enums.Provenance.MODEL_INTERPRETATION`
    accordingly. It is never authoritative, and the system must remain
    fully functional when no such backend is configured.
    """

    def interpret(self, content: TextContent) -> TextContent:
        """Return an interpretation of ``content``, carrying its own provenance."""
        ...


@runtime_checkable
class Clock(Protocol):
    """Supplies the current moment.

    Injected rather than read from the wall clock so that anything
    depending on "now" -- chronology checks against prior records, quiet
    periods -- can be exercised deterministically in tests.
    """

    def now(self) -> dt.datetime:
        """Return the current moment as a timezone-aware timestamp."""
        ...
