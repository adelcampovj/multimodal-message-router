"""Reading the public request-document format.

The format itself is documented in :mod:`message_router.io.schema`.
Standard library only; parsing never evaluates document content.
"""

from __future__ import annotations

from .errors import (
    DocumentParseError,
    FieldError,
    LoadError,
    RelationshipError,
    SchemaVersionError,
    UnknownFieldError,
)
from .loader import LoadedRequest, load_request, load_request_file
from .schema import FORMAT_NAME, SUPPORTED_VERSIONS, UNKNOWN_FIELD_POLICY

__all__ = [
    "load_request",
    "load_request_file",
    "LoadedRequest",
    "FORMAT_NAME",
    "SUPPORTED_VERSIONS",
    "UNKNOWN_FIELD_POLICY",
    "LoadError",
    "DocumentParseError",
    "SchemaVersionError",
    "FieldError",
    "UnknownFieldError",
    "RelationshipError",
]
