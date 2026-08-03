"""Storage-side implementations of the domain ports.

Every query here is recipient-scoped by construction -- see
:mod:`message_router.repositories.history`.
"""

from __future__ import annotations

from .history import InMemoryHistoryRepository

__all__ = ["InMemoryHistoryRepository"]
