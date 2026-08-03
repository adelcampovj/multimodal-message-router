"""Structural guarantees for the routing layer.

These assert *how* routing is built, not what it decides: ordered
first-match with a veto that short-circuits, and no scoring machinery
anywhere that could quietly let positive features offset a risk.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
ROUTING_DIR = SRC_DIR / "message_router" / "routing"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from message_router.routing import (  # noqa: E402
    POLICY_ORDER,
    RoutingContributor,
    RoutingPolicy,
    RoutingResult,
)

ROUTING_SOURCES = sorted(ROUTING_DIR.glob("*.py"))

REQUIRED_POLICY_IDS = (
    "route.safety.hard_veto",
    "route.risk.identity_credential_suppress",
    "route.risk.payment_link_suppress",
    "route.risk.advance_fee_pressure_suppress",
    "route.risk.unusual_channel_suppress",
    "route.risk.coercion_pressure_suppress",
    "route.risk.propagation_low_value_suppress",
    "route.urgency.safe_interrupt",
    "route.value.muted_low_value_suppress",
    "route.value.repetitive_low_value_suppress",
    "route.value.irrelevant_marketing_suppress",
    "route.utility.safe_defer",
    "route.analysis.incomplete_defer",
)

#: Policy identifiers formally replaced in P2C3B. The pressure rule was
#: unreachable -- ``high`` manufactured pressure could only come from a
#: severe finding, which already vetoes -- and the combined payment/identity
#: rule has been split into the explicit families that replaced it.
RETIRED_POLICY_IDS = (
    "route.risk.pressure_suppress",
    "route.risk.payment_identity_suppress",
)


def identifiers(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


class PrecedenceTests(unittest.TestCase):
    def test_policy_order_is_explicit_data(self):
        self.assertIsInstance(POLICY_ORDER, tuple)
        self.assertTrue(all(isinstance(pid, str) and callable(fn) for pid, fn in POLICY_ORDER))

    def test_hard_veto_is_first(self):
        self.assertEqual(POLICY_ORDER[0][0], "route.safety.hard_veto")

    def test_all_required_policy_ids_are_present_in_order(self):
        self.assertEqual([pid for pid, _ in POLICY_ORDER], list(REQUIRED_POLICY_IDS))

    def test_order_is_not_merely_lexical(self):
        ids = [pid for pid, _ in POLICY_ORDER]
        self.assertNotEqual(ids, sorted(ids))
        self.assertNotEqual(ids, sorted(ids, reverse=True))

    def test_policy_ids_are_unique(self):
        ids = [pid for pid, _ in POLICY_ORDER]
        self.assertEqual(len(ids), len(set(ids)))

    def test_retired_policy_ids_leave_no_compatibility_path(self):
        """A replaced rule is gone, not kept as an unreachable alias."""
        ids = {pid for pid, _ in POLICY_ORDER}
        sources = "\n".join(p.read_text(encoding="utf-8") for p in ROUTING_SOURCES)
        for retired in RETIRED_POLICY_IDS:
            self.assertNotIn(retired, ids, retired)
            self.assertNotIn(retired, sources, retired)

    def test_every_suspicious_combination_needs_two_independent_families(self):
        """Each risk rule pairs an anchor with a disjoint corroborator set,
        and at least one side is substantive -- so pressure, a repeated
        modality, or a suspicious-looking domain cannot combine alone."""
        from message_router.analysis import RiskFamily
        from message_router.routing.policy import RISK_COMBINATIONS

        self.assertTrue(RISK_COMBINATIONS)
        for policy_id, anchor, corroborators in RISK_COMBINATIONS:
            self.assertIsInstance(anchor, RiskFamily, policy_id)
            self.assertTrue(corroborators, policy_id)
            self.assertNotIn(anchor, corroborators, policy_id)
            self.assertTrue(
                anchor.is_substantive or all(c.is_substantive for c in corroborators),
                f"{policy_id} can fire on corroboration alone",
            )

    def test_risk_rules_read_projected_families_not_content(self):
        """No routing rule re-derives risk: the families are projected by the
        analysis layer and only read here."""
        source = (ROUTING_DIR / "policy.py").read_text(encoding="utf-8")
        self.assertIn("inputs.features.risk_families", source)
        for term in ("re.compile", "re.search", "\\b", "keyword", "pattern"):
            self.assertNotIn(term, source, f"routing performs its own matching: {term}")

    def test_route_evaluates_rules_in_declared_order(self):
        """Instrumented: record the order predicates are actually called."""
        import message_router.routing.policy as policy_module
        from message_router.analysis import ClassificationResult, DecisionFeatures
        from message_router.safety import SafetyAssessment

        seen: list[str] = []

        def recorder(pid, original):
            def _predicate(inputs):
                seen.append(pid)
                return original(inputs)
            return _predicate

        original = policy_module.POLICY_ORDER
        policy_module.POLICY_ORDER = tuple((pid, recorder(pid, fn)) for pid, fn in original)
        try:
            RoutingPolicy().route(SafetyAssessment(), ClassificationResult(
                selected=__import__("message_router.analysis", fromlist=["MessageClass"]).MessageClass.UNCATEGORIZED
            ), DecisionFeatures())
        finally:
            policy_module.POLICY_ORDER = original
        self.assertEqual(seen, [pid for pid, _ in original][: len(seen)])


class NoWeightedAggregateTests(unittest.TestCase):
    BANNED_IDENTIFIERS = (
        "route_score", "weighted", "weight", "points", "probability", "coefficient",
        "aggregate", "total_score", "confidence", "rationale", "ranking", "similarity",
        "relevance_score", "calibrat",
    )

    def test_no_scoring_identifiers(self):
        for path in ROUTING_SOURCES:
            for name in identifiers(path):
                for term in self.BANNED_IDENTIFIERS:
                    self.assertNotIn(term, name.lower(), f"{path.name}:{name}")

    def test_no_arithmetic_on_feature_levels(self):
        """The only numbers in this package are the explicit level ordering
        used for local comparisons; nothing sums or multiplies them."""
        for path in ROUTING_SOURCES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.BinOp):
                    self.assertNotIsInstance(node.op, (ast.Add, ast.Mult, ast.Sub, ast.Div),
                                             f"{path.name}: arithmetic in routing")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, ("sum", "max", "min"), f"{path.name}: aggregation")

    def test_result_carries_no_score_or_final_outputs(self):
        names = {f.name for f in dataclasses.fields(RoutingResult)}
        names |= {f.name for f in dataclasses.fields(RoutingContributor)}
        for name in names:
            for term in ("score", "weight", "probability", "confidence", "rationale",
                         "reason", "rank", "evidence", "text", "content", "url", "payload"):
                self.assertNotIn(term, name.lower(), name)

    def test_no_final_confidence_or_rationale_construction(self):
        for path in ROUTING_SOURCES:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("Confidence(", source, path.name)
            self.assertNotIn("Rationale(", source, path.name)


class ImmutabilityAndIsolationTests(unittest.TestCase):
    def test_result_models_are_frozen(self):
        for model in (RoutingResult, RoutingContributor):
            self.assertTrue(model.__dataclass_params__.frozen, model.__name__)

    def test_routing_never_constructs_safety_or_analysis_objects(self):
        banned = ("SafetyFinding(", "SafetyAssessment(", "ClassificationResult(",
                  "DecisionFeatures(", "Feature(", "Severity.SEVERE,")
        for path in ROUTING_SOURCES:
            source = path.read_text(encoding="utf-8")
            for term in banned:
                self.assertNotIn(term, source, f"{path.name} constructs {term}")

    def test_route_signature_takes_only_the_three_approved_inputs(self):
        parameters = list(inspect.signature(RoutingPolicy.route).parameters)
        self.assertEqual(parameters, ["self", "assessment", "classification", "features"])


class NoNetworkOrModelTests(unittest.TestCase):
    BANNED = ("urllib", "http.client", "socket", "requests", "httpx", "aiohttp", "openai",
              "anthropic", "transformers", "torch", "sklearn", "numpy", "cv2", "PIL",
              "pytesseract", "speech_recognition", "subprocess", "ssl", "dns", "random", "time")

    def test_no_banned_imports(self):
        for path in ROUTING_SOURCES:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    for banned in self.BANNED:
                        self.assertFalse(module == banned or module.startswith(banned + "."),
                                         f"{path.name} imports {module}")

    def test_no_ambient_clock_or_randomness(self):
        for path in ROUTING_SOURCES:
            source = path.read_text(encoding="utf-8")
            for term in ("datetime.now", "time.time", "random.", "utcnow"):
                self.assertNotIn(term, source, path.name)

    def test_imports_cleanly_with_optional_engines_blocked(self):
        import os

        script = (
            "import sys\n"
            "blocked = {'cv2','PIL','numpy','torch','requests','httpx','openai','anthropic'}\n"
            "class Blocker:\n"
            "    def find_module(self, name, path=None):\n"
            "        if name.split('.')[0] in blocked:\n"
            "            raise ImportError('blocked: ' + name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "import message_router.routing\n"
            "print('IMPORT_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60,
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(SRC_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("IMPORT_OK", result.stdout)


class NoOrganizerVocabularyTests(unittest.TestCase):
    def test_no_frozen_aliases_or_identifiers(self):
        banned = re.compile(
            r"(?i)\b(notify|digest|mute[ds]?)\b|"
            r"\b(?:msg|hist|sample_msg|u|img|vn|group|business)_[0-9]{2,}\b|"
            r"solved[ _-]?(?:example|label)|hackerrank|\borchestrate\b"
        )
        for path in ROUTING_SOURCES:
            matches = banned.findall(path.read_text(encoding="utf-8"))
            self.assertEqual(matches, [], f"{path.name}: {matches}")


class CliSurfaceTests(unittest.TestCase):
    def test_cli_exposes_no_route_or_policy_command(self):
        source = (SRC_DIR / "message_router" / "cli.py").read_text(encoding="utf-8")
        for term in ('"route"', "'route'", '"classify"', "'classify'", '"policy"', "'policy'"):
            self.assertNotIn(term, source)

    def test_cli_does_not_import_routing(self):
        tree = ast.parse((SRC_DIR / "message_router" / "cli.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotIn("routing", node.module)


class ClassificationBindingTests(unittest.TestCase):
    def test_no_action_is_embedded_in_the_classification_module(self):
        """No routing action, by name or by value, appears in classification.

        Matched as whole tokens against every member of the real
        ``RoutingAction`` vocabulary rather than as bare substrings. That
        covers ``defer``, which a substring check had to leave out because
        it collides with ordinary prose, and it stops an unrelated word that
        merely shares a prefix -- a service described as "interrupted" -- from
        reading as a routing action.
        """
        from message_router.domain import RoutingAction

        source = (SRC_DIR / "message_router" / "analysis" / "classification.py").read_text(encoding="utf-8")
        self.assertNotIn("RoutingAction", source)
        for action in RoutingAction:
            found = re.search(rf"\b{action.value}\b", source, re.IGNORECASE)
            self.assertIsNone(found, f"classification references the action {action.value}")
            self.assertNotIn(f"{action.name}", source, action.name)


if __name__ == "__main__":
    unittest.main()
