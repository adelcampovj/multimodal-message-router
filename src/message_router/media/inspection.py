"""Deciding what a file actually is from its leading bytes.

The declared kind in a document and the extension on a filename are both
claims made by whoever supplied the file. This module ignores both and
reads the container signature instead, then reports whether the bytes agree
with what was declared.

Scope is deliberately narrow. Recognising a signature proves the file
*starts* like that format -- nothing more. This is not a decode, not a
validity check, and not a safety check: a file can carry a correct
signature and still be truncated, corrupt, or hostile. Treat a recognised
result as "worth handing to a real decoder", never as "safe".

Standard library only: no decoding, no subprocess, no third-party imaging
or audio library, no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..domain.enums import MediaKind

__all__ = [
    "InspectionStatus",
    "MediaInspection",
    "SIGNATURE_READ_BYTES",
    "inspect_bytes",
    "inspect_file",
]

#: Only this many leading bytes are ever read for signature detection.
#: Every signature below is decided within this window, so an enormous or
#: deliberately padded file cannot make inspection expensive.
SIGNATURE_READ_BYTES = 64

# ISO base media file format brands that indicate audio rather than video.
_AUDIO_BRANDS = frozenset({b"M4A ", b"M4B ", b"mp42", b"isom", b"dash"})


class InspectionStatus(Enum):
    """Outcome of looking at a file's leading bytes."""

    #: A known signature was matched.
    RECOGNIZED = "recognized"
    #: The bytes were read but match no signature this module knows.
    UNKNOWN_FORMAT = "unknown_format"
    #: The bytes could not be read at all.
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class MediaInspection:
    """What the bytes say, next to what the document claimed.

    Carries no path: a result may be logged or attached to a report, and a
    local filesystem location is not something that should travel with it.
    """

    media_id: str
    declared_kind: MediaKind
    status: InspectionStatus
    detected_family: MediaKind | None = None
    detected_format: str | None = None
    byte_size: int | None = None

    @property
    def is_recognized(self) -> bool:
        return self.status is InspectionStatus.RECOGNIZED

    @property
    def matches_declared(self) -> bool:
        """True only when the bytes were recognised *and* agree with the claim.

        Unrecognised and unreadable both return False: neither is evidence
        that the declaration was correct.
        """
        return self.detected_family is not None and self.detected_family is self.declared_kind

    @property
    def kind_mismatch(self) -> bool:
        """True only for a positive contradiction.

        Distinct from ``not matches_declared``: this is reserved for the
        case where the bytes were recognised and say something *different*
        from the declaration, rather than merely failing to confirm it.
        """
        return self.detected_family is not None and self.detected_family is not self.declared_kind


def _detect(head: bytes) -> tuple[MediaKind, str] | None:
    """Return ``(family, format)`` for a recognised signature, else None."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return MediaKind.IMAGE, "png"
    if head.startswith(b"\xff\xd8\xff"):
        return MediaKind.IMAGE, "jpeg"
    if head.startswith(b"RIFF") and len(head) >= 12:
        container = head[8:12]
        if container == b"WEBP":
            return MediaKind.IMAGE, "webp"
        if container == b"WAVE":
            return MediaKind.AUDIO, "wav"
        return None
    if head.startswith(b"ID3"):
        return MediaKind.AUDIO, "mp3"
    # MPEG audio frame sync: eleven set bits opening the frame header.
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return MediaKind.AUDIO, "mp3"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        if head[8:12] in _AUDIO_BRANDS:
            return MediaKind.AUDIO, "iso-bmff"
        # A container this module cannot confidently call audio is left
        # unrecognised rather than guessed at.
        return None
    return None


def inspect_bytes(media_id: str, declared_kind: MediaKind, head: bytes, byte_size: int | None = None) -> MediaInspection:
    """Classify an already-read byte prefix.

    Split out from :func:`inspect_file` so signature logic can be exercised
    without touching a filesystem.
    """
    detected = _detect(head[:SIGNATURE_READ_BYTES])
    if detected is None:
        return MediaInspection(
            media_id=media_id,
            declared_kind=declared_kind,
            status=InspectionStatus.UNKNOWN_FORMAT,
            byte_size=byte_size,
        )
    family, fmt = detected
    return MediaInspection(
        media_id=media_id,
        declared_kind=declared_kind,
        status=InspectionStatus.RECOGNIZED,
        detected_family=family,
        detected_format=fmt,
        byte_size=byte_size,
    )


def inspect_file(media_id: str, declared_kind: MediaKind, path: Path) -> MediaInspection:
    """Read at most :data:`SIGNATURE_READ_BYTES` from ``path`` and classify it.

    An unreadable file produces an
    :attr:`~InspectionStatus.UNREADABLE` result rather than raising, so a
    permission problem or a vanished file is an ordinary reportable
    outcome. The failing path never appears in the result.
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            head = handle.read(SIGNATURE_READ_BYTES)
    except OSError:
        return MediaInspection(
            media_id=media_id,
            declared_kind=declared_kind,
            status=InspectionStatus.UNREADABLE,
        )
    return inspect_bytes(media_id, declared_kind, head, byte_size=size)
