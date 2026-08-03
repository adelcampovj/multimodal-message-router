"""Dispatching an attachment to whichever backends apply to it.

The coordinator owns no engine. It decides which *roles* are applicable to
an attachment's kind, asks whatever backends were injected, and normalises
whatever comes back -- including exceptions -- into the controlled states
the domain already defines.

Five outcomes are kept distinct for every role, because collapsing them
loses the difference between "we could not look" and "we looked and there
was nothing":

* **not attempted** -- the role does not apply to this kind of attachment;
* **unavailable** -- the role applies but no backend was supplied;
* **completed, no content** -- a backend ran and found nothing;
* **completed, content** -- a backend ran and returned something;
* **failed** -- a backend ran and broke.

It makes no judgement about what any returned content *means*. There is no
routing action, category, risk, trust, evidence score, reason, or
confidence produced anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..domain.enums import ExtractionStatus, MediaKind, Provenance
from ..domain.media import MediaAnalysis, MediaReference

__all__ = ["AnalysisRole", "RoleResult", "MediaCoordinator"]


class AnalysisRole(Enum):
    """A kind of reading that can be performed on an attachment."""

    OCR = "ocr"
    ASR = "asr"
    QR = "qr"

    @property
    def applies_to(self) -> MediaKind:
        """The one attachment kind this role can be performed on."""
        return _ROLE_KIND[self]


_ROLE_KIND = {
    AnalysisRole.OCR: MediaKind.IMAGE,
    AnalysisRole.QR: MediaKind.IMAGE,
    AnalysisRole.ASR: MediaKind.AUDIO,
}

_ROLE_PROVENANCE = {
    AnalysisRole.OCR: Provenance.OCR_TEXT,
    AnalysisRole.QR: Provenance.QR_DERIVED,
    AnalysisRole.ASR: Provenance.ASR_TRANSCRIPT,
}


@dataclass(frozen=True)
class RoleResult:
    """One role's outcome for one attachment.

    The role is recorded alongside the analysis so a caller can tell which
    of several identical-looking empty results came from which reading.
    """

    role: AnalysisRole
    analysis: MediaAnalysis

    @property
    def produced_content(self) -> bool:
        return self.analysis.produced_content


class MediaCoordinator:
    """Runs the applicable readings for an attachment.

    Backends are injected and entirely optional; omitting one is a normal
    configuration, reported as
    :attr:`~message_router.domain.enums.ExtractionStatus.BACKEND_UNAVAILABLE`
    rather than treated as an error. Nothing here imports an engine, an
    operating-system API, or a network client.
    """

    def __init__(self, *, ocr=None, asr=None, qr=None) -> None:
        self._backends = {
            AnalysisRole.OCR: ocr,
            AnalysisRole.ASR: asr,
            AnalysisRole.QR: qr,
        }

    def analyze(self, media: MediaReference) -> tuple[RoleResult, ...]:
        """Return one result per role, in a fixed order.

        Every role is always reported, so an absent reading is visible as a
        stated outcome instead of a missing entry the caller has to notice.
        """
        return tuple(self._run_role(role, media) for role in AnalysisRole)

    # -- internals --------------------------------------------------------

    def _run_role(self, role: AnalysisRole, media: MediaReference) -> RoleResult:
        if media.kind is not role.applies_to:
            return RoleResult(role, MediaAnalysis(media, ExtractionStatus.NOT_ATTEMPTED))

        backend = self._backends[role]
        if backend is None:
            return RoleResult(role, MediaAnalysis(media, ExtractionStatus.BACKEND_UNAVAILABLE))

        try:
            analysis = self._invoke(role, backend, media)
        except Exception:
            # Deliberately broad, and deliberately silent about the cause: a
            # third-party backend's exception text is outside this project's
            # control and could carry a local path or a fragment of the
            # content it choked on. The failure is reported; its message is
            # not propagated.
            return RoleResult(role, MediaAnalysis(media, ExtractionStatus.FAILED))

        return RoleResult(role, self._verify(role, media, analysis))

    @staticmethod
    def _invoke(role: AnalysisRole, backend, media: MediaReference) -> MediaAnalysis:
        if role is AnalysisRole.OCR:
            return backend.extract_text(media)
        if role is AnalysisRole.ASR:
            return backend.transcribe(media)
        return backend.decode(media)

    @staticmethod
    def _verify(role: AnalysisRole, media: MediaReference, analysis: object) -> MediaAnalysis:
        """Hold a backend to the contract instead of trusting its return value.

        A backend is replaceable and possibly third-party, so its output is
        checked: wrong type, wrong attachment, or content tagged with a
        provenance that does not belong to this role all become a controlled
        failure rather than propagating into the domain.
        """
        if not isinstance(analysis, MediaAnalysis):
            return MediaAnalysis(media, ExtractionStatus.FAILED)
        if analysis.media != media:
            return MediaAnalysis(media, ExtractionStatus.FAILED)
        if analysis.content is not None and analysis.content.provenance is not _ROLE_PROVENANCE[role]:
            return MediaAnalysis(media, ExtractionStatus.FAILED)
        return analysis
