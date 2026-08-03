"""Ordered, first-match routing policy.

Selects exactly one action from an explicit precedence list. Produces no
rationale, confidence, evidence ranking, probability, or aggregate score.

Standard library only.
"""

from __future__ import annotations

from .models import ContributorEffect, RoutingContributor, RoutingDiagnostic, RoutingResult
from .policy import POLICY_ORDER, RoutingPolicy, route

__all__ = [
    "RoutingPolicy",
    "route",
    "RoutingResult",
    "RoutingContributor",
    "ContributorEffect",
    "RoutingDiagnostic",
    "POLICY_ORDER",
]
