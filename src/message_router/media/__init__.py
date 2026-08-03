"""Attachment handling: safe resolution, byte inspection, backend dispatch.

Standard library only. Importing this package requires no imaging library,
no speech engine, and no operating-system media API.
"""

from __future__ import annotations

from .coordinator import AnalysisRole, MediaCoordinator, RoleResult
from .inspection import (
    SIGNATURE_READ_BYTES,
    InspectionStatus,
    MediaInspection,
    inspect_bytes,
    inspect_file,
)
from .paths import MediaPathError, MediaRootResolver, PathRejection

__all__ = [
    "MediaRootResolver",
    "MediaPathError",
    "PathRejection",
    "MediaInspection",
    "InspectionStatus",
    "inspect_bytes",
    "inspect_file",
    "SIGNATURE_READ_BYTES",
    "MediaCoordinator",
    "AnalysisRole",
    "RoleResult",
]
