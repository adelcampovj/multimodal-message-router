"""Attachment references and the results of asking a backend to read them.

Contains no file handling, no decoding, and no engine selection. These
types describe *what a backend reported*, so the rest of the domain can
reason about attachments without depending on any particular engine being
installed.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..validation.invariants import require_identifier, require_no_embedded_path, require_unit_interval
from .content import TextContent
from .enums import (
    AnalysisWarning,
    ExtractionStatus,
    MediaKind,
    Provenance,
    Reliability,
)
from .errors import ValidationError, ValidationIssue

__all__ = [
    "MediaReference",
    "BackendIdentity",
    "MediaAnalysis",
    "PROVENANCE_REQUIRES_KIND",
]

#: Which attachment kind each extracted provenance may legitimately come
#: from. Text read off audio, or a code decoded from a sound file, means
#: something upstream is mislabelled -- so the pairing is checked.
PROVENANCE_REQUIRES_KIND: dict[Provenance, MediaKind] = {
    Provenance.OCR_TEXT: MediaKind.IMAGE,
    Provenance.QR_DERIVED: MediaKind.IMAGE,
    Provenance.ASR_TRANSCRIPT: MediaKind.AUDIO,
}


@dataclass(frozen=True)
class MediaReference:
    """A stable pointer to an attachment, with no filesystem detail.

    Holds an opaque identifier and a declared kind. It deliberately carries
    no path, URL, or handle: resolving an identifier to bytes is a backend's
    job, and keeping location out of the domain keeps machine-specific
    detail out of every object that quotes one.
    """

    media_id: str
    kind: MediaKind

    def __post_init__(self) -> None:
        object.__setattr__(self, "media_id", require_identifier(self.media_id, "media_id"))
        if not isinstance(self.kind, MediaKind):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "kind")


@dataclass(frozen=True)
class BackendIdentity:
    """Which engine produced a result, named without disclosing where it lives.

    ``name`` is rejected if it looks like a path, so an implementation
    cannot casually record an executable location that would then travel
    with every result it produced.
    """

    name: str
    version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_no_embedded_path(self.name, "name"))
        if self.version is not None:
            object.__setattr__(self, "version", require_no_embedded_path(self.version, "version"))


@dataclass(frozen=True)
class MediaAnalysis:
    """What a backend reported about one attachment.

    The status/content pairing is enforced rather than assumed:

    * :attr:`~message_router.domain.enums.ExtractionStatus.SUCCEEDED`
      requires ``content``;
    * every other status requires ``content`` to be ``None``.

    That keeps "the engine was missing", "the engine found nothing", and
    "the engine failed" from all collapsing into an indistinguishable empty
    string. When content is present, its provenance must match the
    attachment kind (see :data:`PROVENANCE_REQUIRES_KIND`).
    """

    media: MediaReference
    status: ExtractionStatus
    backend: BackendIdentity | None = None
    content: TextContent | None = None
    reliability: Reliability = Reliability.MEDIUM
    reported_score: float | None = None
    warnings: tuple[AnalysisWarning, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.media, MediaReference):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "media")
        if not isinstance(self.status, ExtractionStatus):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "status")
        if self.backend is not None and not isinstance(self.backend, BackendIdentity):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "backend")
        if not isinstance(self.reliability, Reliability):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "reliability")

        if self.status.yields_content:
            if not isinstance(self.content, TextContent):
                raise ValidationError(ValidationIssue.STATUS_CONTENT_MISMATCH, "content")
        elif self.content is not None:
            raise ValidationError(ValidationIssue.STATUS_CONTENT_MISMATCH, "content")

        if isinstance(self.content, TextContent):
            required = PROVENANCE_REQUIRES_KIND.get(self.content.provenance)
            if required is None:
                raise ValidationError(ValidationIssue.PROVENANCE_MISMATCH, "content")
            if required is not self.media.kind:
                raise ValidationError(ValidationIssue.MEDIA_KIND_MISMATCH, "content")

        if self.reported_score is not None:
            object.__setattr__(
                self, "reported_score", require_unit_interval(self.reported_score, "reported_score")
            )
        object.__setattr__(self, "warnings", tuple(self.warnings))
        for warning in self.warnings:
            if not isinstance(warning, AnalysisWarning):
                raise ValidationError(ValidationIssue.WRONG_TYPE, "warnings")

    @property
    def produced_content(self) -> bool:
        """True only when a backend actually returned text."""
        return self.content is not None

    @property
    def was_attempted(self) -> bool:
        """False when no backend was ever asked to read this attachment."""
        return self.status is not ExtractionStatus.NOT_ATTEMPTED
