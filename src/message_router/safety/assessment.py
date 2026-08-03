"""Running the rules across every available source and aggregating.

The analyzer takes validated domain objects, evaluates each provenance-
tagged piece of content separately, records what each source was able to
contribute, notes disagreements between sources, and returns one immutable
:class:`~message_router.safety.models.SafetyAssessment`.

Three properties this layer is built around:

* **Sources never merge.** Each piece of content is analysed on its own and
  every finding keeps the provenance it came from. Concatenating a caption
  with a transcript would invent adjacency that never existed.
* **Absence is never safety.** A backend that was missing, failed, or
  returned nothing produces a recorded gap and marks the analysis
  incomplete. It never removes a finding and never lowers a severity.
* **Engagement is not an input.** Preferences, history, quiet hours, and
  silenced-conversation state are not parameters of this function. A
  well-liked sender and an ignored thread get identical safety analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.content import TextContent
from ..domain.enums import ExtractionStatus, Provenance, Reliability
from ..domain.errors import ValidationError, ValidationIssue
from ..domain.media import MediaAnalysis
from ..domain.message import IncomingMessage
from .models import (
    AUTHORED_SOURCE_REF,
    ConflictKind,
    DiagnosticCode,
    ModalityConflict,
    ModalityOutcome,
    ModalityState,
    SafetyAssessment,
    SafetyFinding,
    Severity,
    cap_reliability,
)
from .normalization import NormalizedText, normalize
from .rules import evaluate_content, evaluate_propagation

__all__ = ["SafetyAnalyzer", "analyze_message"]

_STATUS_TO_STATE = {
    ExtractionStatus.SUCCEEDED: ModalityState.ANALYZED,
    ExtractionStatus.NO_CONTENT_FOUND: ModalityState.EMPTY,
    ExtractionStatus.BACKEND_UNAVAILABLE: ModalityState.UNAVAILABLE,
    ExtractionStatus.FAILED: ModalityState.FAILED,
    ExtractionStatus.NOT_ATTEMPTED: ModalityState.NOT_APPLICABLE,
}

#: Categories whose presence, combined with coercive payment, is enough to
#: escalate that payment finding to severe. Each is an independent signal;
#: the escalation is the conjunction, never either one alone.
_ESCALATING_CATEGORIES = ("DECEPTIVE_LINK", "IMPERSONATION")

_MIN_COMPARABLE_TOKENS = 4


@dataclass(frozen=True)
class _Piece:
    """One provenance-tagged piece of content queued for analysis."""

    content: TextContent
    source_ref: str
    normalized: NormalizedText


class SafetyAnalyzer:
    """Deterministic safety analysis over one message and its attachments.

    Stateless and side-effect free: the same inputs always produce an equal
    assessment, and nothing is written, logged, or fetched.
    """

    def analyze(
        self,
        message: IncomingMessage,
        media_analyses: tuple[MediaAnalysis, ...] = (),
        forwarding_count: int | None = None,
    ) -> SafetyAssessment:
        if not isinstance(message, IncomingMessage):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "message")
        if forwarding_count is not None and (
            isinstance(forwarding_count, bool) or not isinstance(forwarding_count, int)
        ):
            raise ValidationError(ValidationIssue.WRONG_TYPE, "forwarding_count")

        diagnostics: list[DiagnosticCode] = []
        outcomes: list[ModalityOutcome] = []
        pieces: list[_Piece] = []

        # -- authored text ------------------------------------------------
        if message.body is None:
            diagnostics.append(DiagnosticCode.NO_AUTHORED_BODY)
            outcomes.append(
                ModalityOutcome(
                    Provenance.AUTHORED_TEXT,
                    AUTHORED_SOURCE_REF,
                    ModalityState.NOT_APPLICABLE,
                    Reliability.HIGH,
                )
            )
        else:
            normalized = normalize(message.body.value)
            if normalized.truncated:
                diagnostics.append(DiagnosticCode.CONTENT_TRUNCATED)
            state = ModalityState.EMPTY if normalized.is_empty else ModalityState.ANALYZED
            outcomes.append(
                ModalityOutcome(Provenance.AUTHORED_TEXT, AUTHORED_SOURCE_REF, state, Reliability.HIGH)
            )
            if state is ModalityState.ANALYZED:
                pieces.append(_Piece(message.body, AUTHORED_SOURCE_REF, normalized))

        # -- extracted content --------------------------------------------
        for analysis in media_analyses:
            if not isinstance(analysis, MediaAnalysis):
                raise ValidationError(ValidationIssue.WRONG_TYPE, "media_analyses")
            source_ref = analysis.media.media_id
            state = _STATUS_TO_STATE.get(analysis.status, ModalityState.FAILED)
            content = analysis.content

            if content is None:
                outcomes.append(
                    ModalityOutcome(
                        _provenance_for(analysis), source_ref, state, cap_reliability(
                            Reliability.MEDIUM, _provenance_for(analysis)
                        )
                    )
                )
                continue

            normalized = normalize(content.value)
            if normalized.truncated:
                diagnostics.append(DiagnosticCode.CONTENT_TRUNCATED)
            if normalized.is_empty:
                state = ModalityState.EMPTY
            outcomes.append(
                ModalityOutcome(
                    content.provenance,
                    source_ref,
                    state,
                    cap_reliability(Reliability.MEDIUM, content.provenance),
                )
            )
            if state is ModalityState.ANALYZED:
                pieces.append(_Piece(content, source_ref, normalized))

        # -- rules ---------------------------------------------------------
        has_asserted_trust = bool(message.sender is not None and message.sender.has_asserted_trust)
        findings: list[SafetyFinding] = []
        for piece in pieces:
            for hit in evaluate_content(piece.normalized, has_asserted_trust=has_asserted_trust):
                findings.append(
                    SafetyFinding(
                        category=hit.category,
                        severity=hit.severity,
                        rule_id=hit.rule_id,
                        provenance=piece.content.provenance,
                        source_ref=piece.source_ref,
                        diagnostic=hit.diagnostic,
                        reliability=cap_reliability(hit.reliability, piece.content.provenance),
                        ambiguous=hit.ambiguous,
                        indicators=hit.indicators,
                    )
                )

        authored_piece = next(
            (p for p in pieces if p.content.provenance is Provenance.AUTHORED_TEXT), None
        )
        for hit in evaluate_propagation(
            authored_piece.normalized if authored_piece else None, forwarding_count
        ):
            findings.append(
                SafetyFinding(
                    category=hit.category,
                    severity=hit.severity,
                    rule_id=hit.rule_id,
                    provenance=(
                        authored_piece.content.provenance
                        if authored_piece and hit.rule_id != "PROP_HIGH_FORWARD_COUNT"
                        else Provenance.STRUCTURED_METADATA
                    ),
                    source_ref=(
                        authored_piece.source_ref
                        if authored_piece and hit.rule_id != "PROP_HIGH_FORWARD_COUNT"
                        else "structured"
                    ),
                    diagnostic=hit.diagnostic,
                    reliability=hit.reliability,
                    ambiguous=hit.ambiguous,
                )
            )

        findings = self._escalate_combinations(findings)

        # -- conflicts and completeness ------------------------------------
        conflicts = self._detect_conflicts(pieces)
        for conflict in conflicts:
            diagnostics.append(conflict.diagnostic)

        degraded = [o for o in outcomes if o.state.is_degraded]
        incomplete = bool(degraded) or any(
            d is DiagnosticCode.CONTENT_TRUNCATED for d in diagnostics
        )
        if incomplete:
            diagnostics.append(DiagnosticCode.ANALYSIS_INCOMPLETE)

        return SafetyAssessment(
            findings=tuple(sorted(findings, key=lambda f: f.sort_key())),
            modality_outcomes=tuple(sorted(outcomes, key=lambda o: o.sort_key())),
            modality_conflicts=tuple(sorted(conflicts, key=lambda c: c.sort_key())),
            diagnostics=tuple(sorted({d for d in diagnostics}, key=lambda d: d.value)),
            analysis_incomplete=incomplete,
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _escalate_combinations(findings: list[SafetyFinding]) -> list[SafetyFinding]:
        """Raise coercive payment to severe when corroborated.

        Payment pressure alone is suspicious; payment pressure alongside a
        deceptive link or an unsupported identity claim is the shape of an
        actual scam. Escalation only ever moves upward, and only for the
        payment finding -- the corroborating findings keep their own
        severity.
        """
        present = {f.category.name for f in findings}
        if not (present & set(_ESCALATING_CATEGORIES)):
            return findings
        escalated: list[SafetyFinding] = []
        for finding in findings:
            if (
                finding.category.name == "COERCIVE_PAYMENT"
                and finding.severity is Severity.SUSPICIOUS
                and not finding.ambiguous
            ):
                escalated.append(
                    SafetyFinding(
                        category=finding.category,
                        severity=Severity.SEVERE,
                        rule_id=finding.rule_id,
                        provenance=finding.provenance,
                        source_ref=finding.source_ref,
                        diagnostic=finding.diagnostic,
                        reliability=finding.reliability,
                        ambiguous=False,
                        indicators=finding.indicators,
                    )
                )
            else:
                escalated.append(finding)
        return escalated

    @staticmethod
    def _detect_conflicts(pieces: list[_Piece]) -> list[ModalityConflict]:
        """Report disagreement between sources without ranking them.

        "Disagreement" here is deliberately coarse -- entirely disjoint
        vocabulary between two sources describing the same message. It
        flags a reason for a later stage to be less certain; it never
        decides which source is right.
        """
        conflicts: list[ModalityConflict] = []
        if len(pieces) < 2:
            return conflicts

        token_sets = []
        for piece in pieces:
            # The security view, so a token split by an invisible character
            # in one modality still compares equal to the same word in another.
            tokens = {t for t in piece.normalized.security.split() if len(t) > 3}
            token_sets.append((piece, tokens))

        for index, (left, left_tokens) in enumerate(token_sets):
            for right, right_tokens in token_sets[index + 1 :]:
                if len(left_tokens) < _MIN_COMPARABLE_TOKENS or len(right_tokens) < _MIN_COMPARABLE_TOKENS:
                    conflicts.append(
                        ModalityConflict(
                            ConflictKind.INSUFFICIENT_COMPARISON,
                            left.content.provenance,
                            right.content.provenance,
                            DiagnosticCode.INSUFFICIENT_COMPARISON,
                        )
                    )
                    continue
                if left_tokens & right_tokens:
                    continue
                authored = Provenance.AUTHORED_TEXT
                kind = (
                    ConflictKind.AUTHORED_VS_EXTRACTED
                    if authored in (left.content.provenance, right.content.provenance)
                    else ConflictKind.EXTRACTED_VS_EXTRACTED
                )
                diagnostic = (
                    DiagnosticCode.AUTHORED_EXTRACTED_CONFLICT
                    if kind is ConflictKind.AUTHORED_VS_EXTRACTED
                    else DiagnosticCode.EXTRACTED_EXTRACTED_CONFLICT
                )
                conflicts.append(
                    ModalityConflict(
                        kind, left.content.provenance, right.content.provenance, diagnostic
                    )
                )
        return conflicts


def _provenance_for(analysis: MediaAnalysis) -> Provenance:
    """Best available provenance for an analysis that returned no content."""
    if analysis.content is not None:
        return analysis.content.provenance
    from ..domain.enums import MediaKind

    return (
        Provenance.OCR_TEXT
        if analysis.media.kind is MediaKind.IMAGE
        else Provenance.ASR_TRANSCRIPT
        if analysis.media.kind is MediaKind.AUDIO
        else Provenance.STRUCTURED_METADATA
    )


def analyze_message(
    message: IncomingMessage,
    media_analyses: tuple[MediaAnalysis, ...] = (),
    forwarding_count: int | None = None,
) -> SafetyAssessment:
    """Convenience wrapper around :class:`SafetyAnalyzer`."""
    return SafetyAnalyzer().analyze(message, media_analyses, forwarding_count)
