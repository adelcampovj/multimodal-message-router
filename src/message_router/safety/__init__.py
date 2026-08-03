"""Deterministic, dependency-free safety analysis.

Consumes validated domain objects and returns structured findings plus an
aggregate :class:`SafetyAssessment`. It reports risk; it does not route.
Nothing here produces a routing action, a message category, a rationale, a
confidence value, or an evidence ranking.

Standard library only: no model, no network, no optional media engine.
"""

from __future__ import annotations

from .assessment import SafetyAnalyzer, analyze_message
from .models import (
    ConflictKind,
    DiagnosticCode,
    FindingCategory,
    ModalityConflict,
    ModalityOutcome,
    ModalityState,
    SafetyAssessment,
    SafetyFinding,
    Severity,
)

__all__ = [
    "SafetyAnalyzer",
    "analyze_message",
    "SafetyAssessment",
    "SafetyFinding",
    "Severity",
    "FindingCategory",
    "DiagnosticCode",
    "ModalityState",
    "ModalityOutcome",
    "ModalityConflict",
    "ConflictKind",
]
