"""The independent domain contracts this project is built on.

This is the intended stable surface. Import from here rather than from the
individual submodules, which may be reorganized.

Nothing in this package makes a routing decision, detects risk, ranks
history, or reads a file. It defines the vocabulary and the invariants that
such components will later be held to.
"""

from __future__ import annotations

from .content import DeterministicSignal, RiskSignal, TextContent, TrustSignal
from .context import (
    ContextGap,
    ConversationContext,
    InteractionRecord,
    QuietPeriod,
    RecipientPreferences,
    RecipientScope,
    SenderContext,
)
from .decision import Confidence, Rationale, RoutingDecision
from .enums import (
    AbsenceReason,
    AnalysisWarning,
    ConversationSurface,
    ExtractionStatus,
    MediaKind,
    MessageCategory,
    Provenance,
    Reliability,
    RiskCategory,
    RoutingAction,
    TrustCategory,
)
from .errors import DomainError, UnsupportedValueError, ValidationError, ValidationIssue
from .evidence import EvidenceReference
from .media import BackendIdentity, MediaAnalysis, MediaReference
from .message import IncomingMessage

__all__ = [
    # Vocabularies
    "RoutingAction",
    "MessageCategory",
    "Provenance",
    "MediaKind",
    "ExtractionStatus",
    "Reliability",
    "ConversationSurface",
    "AbsenceReason",
    "AnalysisWarning",
    "RiskCategory",
    "TrustCategory",
    # Content and signals
    "TextContent",
    "DeterministicSignal",
    "RiskSignal",
    "TrustSignal",
    # Parties and surroundings
    "RecipientScope",
    "SenderContext",
    "ConversationContext",
    "QuietPeriod",
    "RecipientPreferences",
    "InteractionRecord",
    "ContextGap",
    # Message and attachments
    "IncomingMessage",
    "MediaReference",
    "MediaAnalysis",
    "BackendIdentity",
    # Background records
    "EvidenceReference",
    # Outcome
    "RoutingDecision",
    "Confidence",
    "Rationale",
    # Errors
    "DomainError",
    "ValidationError",
    "ValidationIssue",
    "UnsupportedValueError",
]
