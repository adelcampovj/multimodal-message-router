"""One request in, one routing decision out, through the existing stages.

This module adds no rule of its own. It calls the same four public stages in
the same order that the test suite's end-to-end helper does --
safety analysis, classification, feature extraction, ordered routing -- and
renders the result using the serialization conventions the stages already
have. Anything that decides, scores or classifies lives in those stages.

Two things are deliberately *not* done here:

* No recognition. An attachment is handed to a
  :class:`~message_router.media.MediaCoordinator`; unless the caller supplies
  backends, every reading is reported as unavailable and the decision says
  its analysis is incomplete. Nothing here reads an image or transcribes
  audio.
* No fabrication. The rendered decision contains only values the stages
  produced: no confidence, no reason text, no score, and no content from the
  request.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass

from .analysis import ClassificationResult, DecisionFeatures, classify, extract_features
from .io import LoadedRequest
from .media import MediaCoordinator
from .routing import RoutingResult, route
from .safety import SafetyAssessment, analyze_message

__all__ = ["RoutedRequest", "route_request", "decision_to_dict"]


@dataclass(frozen=True)
class RoutedRequest:
    """Every stage's output for one request, kept so none is re-derived."""

    assessment: SafetyAssessment
    classification: ClassificationResult
    features: DecisionFeatures
    result: RoutingResult


def route_request(
    loaded: LoadedRequest, *, coordinator: MediaCoordinator | None = None
) -> RoutedRequest:
    """Route one loaded request through the existing stages.

    ``coordinator`` supplies attachment readings. When it is omitted, a
    coordinator with no backends is used, which is the honest default: the
    attachment is acknowledged and reported as unread.
    """
    message = loaded.message

    media_analyses = ()
    if message.attachment is not None:
        reader = coordinator if coordinator is not None else MediaCoordinator()
        media_analyses = tuple(role.analysis for role in reader.analyze(message.attachment))

    assessment = analyze_message(message, media_analyses=media_analyses)

    # Every text the stages may read, each paired with where it came from.
    texts = []
    if message.body is not None:
        texts.append((message.body.value, message.body.provenance))
    for analysis in media_analyses:
        if analysis.content is not None:
            texts.append((analysis.content.value, analysis.content.provenance))

    classification = classify(tuple(texts), assessment)
    features = extract_features(
        message, assessment, classification, preferences=loaded.preferences
    )
    return RoutedRequest(assessment, classification, features, route(assessment, classification, features))


def decision_to_dict(routed: RoutedRequest) -> dict:
    """A JSON-ready, content-free view of one routed request.

    The routing fields are exactly those of
    :meth:`~message_router.routing.RoutingResult.serialize`, under the same
    names. Classification and safety findings are added from the stages that
    produced them. Finding provenance is what lets a reader see that a
    decision came from text read out of an image or a recording.
    """
    decision = dict(routed.result.serialize())
    decision["contributors"] = [
        {"family": family, "level": level, "effect": effect, "rule_ids": list(rule_ids)}
        for family, level, effect, rule_ids in decision["contributors"]
    ]
    decision["diagnostics"] = list(decision["diagnostics"])

    decision["classification"] = {
        "selected": routed.classification.selected.value,
        "ambiguous": routed.classification.ambiguous,
    }

    highest = routed.assessment.highest_severity
    decision["safety"] = {
        "highest_severity": highest.value if highest is not None else None,
        "findings": [
            {
                "rule_id": finding.rule_id,
                "category": finding.category.value,
                "severity": finding.severity.value,
                "provenance": finding.provenance.value,
            }
            for finding in sorted(routed.assessment.findings, key=lambda f: f.sort_key())
        ],
    }
    return decision
