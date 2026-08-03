"""Structural invariant checks shared across the domain layer.

Import the individual checks from :mod:`message_router.validation.invariants`.
These verify shape, not safety -- see that module's docstring.
"""

from __future__ import annotations

from .invariants import (
    TIMESTAMP_POLICY,
    normalize_timestamp,
    require_aware_timestamp,
    require_identifier,
    require_no_embedded_path,
    require_strictly_prior,
    require_text,
    require_unit_interval,
)

__all__ = [
    "TIMESTAMP_POLICY",
    "normalize_timestamp",
    "require_aware_timestamp",
    "require_identifier",
    "require_no_embedded_path",
    "require_strictly_prior",
    "require_text",
    "require_unit_interval",
]
