"""Independent classification and routing-relevant feature extraction.

Describes what a message is and what a later routing stage would need to
know about it. Chooses no route: nothing here produces a routing action,
threshold, precedence rule, confidence value, rationale, or evidence
ranking.

Standard library only.
"""

from __future__ import annotations

from .classification import (
    ClassificationContributor,
    ClassificationDiagnostic,
    ClassificationResult,
    MessageClass,
    SupportStrength,
    classify,
)
from .features import (
    Completeness,
    ContentAvailability,
    DecisionFeatures,
    Feature,
    FeatureContributor,
    FeatureFamily,
    FeatureLevel,
    FeatureNote,
    InteractionSummary,
    ModalityConsistency,
    RiskFamily,
    RiskRootCause,
    SupportSource,
    extract_features,
)

__all__ = [
    "classify",
    "MessageClass",
    "ClassificationResult",
    "ClassificationContributor",
    "ClassificationDiagnostic",
    "SupportStrength",
    "extract_features",
    "DecisionFeatures",
    "Feature",
    "FeatureContributor",
    "FeatureFamily",
    "FeatureLevel",
    "FeatureNote",
    "SupportSource",
    "ModalityConsistency",
    "Completeness",
    "ContentAvailability",
    "RiskFamily",
    "RiskRootCause",
    "InteractionSummary",
]
