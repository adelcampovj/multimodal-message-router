"""Replaceable backend interfaces.

Structural Protocols only -- see :mod:`message_router.ports.backends`.
Importing this package pulls in nothing beyond the standard library and the
domain contracts.
"""

from __future__ import annotations

from .backends import (
    AsrBackend,
    Clock,
    EvidenceRepository,
    OcrBackend,
    QrBackend,
    SemanticAnalysisBackend,
)

__all__ = [
    "OcrBackend",
    "AsrBackend",
    "QrBackend",
    "EvidenceRepository",
    "SemanticAnalysisBackend",
    "Clock",
]
