"""Structural guarantees for the analysis layer.

These assert the layer stays inside its remit: it describes a message and
the inputs a routing decision would need, and it decides nothing.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
ANALYSIS_DIR = SRC_DIR / "message_router" / "analysis"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from message_router.analysis import (  # noqa: E402
    ClassificationContributor,
    ClassificationResult,
    DecisionFeatures,
    Feature,
    FeatureContributor,
    MessageClass,
    classify,
    extract_features,
)

ANALYSIS_SOURCES = sorted(ANALYSIS_DIR.glob("*.py"))


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


class NoRoutingDecisionTests(unittest.TestCase):
    def test_no_routing_or_decision_construction(self):
        banned = ("RoutingDecision(", "RoutingAction.", "Confidence(", "Rationale(", "MessageCategory.")
        for path in ANALYSIS_SOURCES:
            source = path.read_text(encoding="utf-8")
            for term in banned:
                self.assertNotIn(term, source, f"{path.name} references {term}")

    def test_no_import_of_routing_or_decision_modules(self):
        for path in ANALYSIS_SOURCES:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertNotIn("routing", node.module, path.name)
                    self.assertNotIn("decision", node.module, path.name)

    def test_classifier_signature_accepts_no_routing_action(self):
        parameters = inspect.signature(classify).parameters
        for name in parameters:
            for term in ("action", "route", "confidence", "rationale"):
                self.assertNotIn(term, name.lower())

    def test_feature_extraction_signature_accepts_no_routing_action(self):
        """Matches whole words: `interaction_summary` legitimately contains
        the substring "action" without being a routing action."""
        import re

        banned = re.compile(r"(?:^|_)(?:action|route|routing|threshold|precedence)(?:_|$)")
        for name in inspect.signature(extract_features).parameters:
            self.assertIsNone(banned.search(name.lower()), name)

    def test_no_threshold_or_precedence_identifiers(self):
        """Bans route thresholds and precedence.

        Two threshold constants are permitted and named explicitly: the
        forwarding-metadata cut-offs, which describe an input, not a route.
        """
        allowed = {"FORWARD_METADATA_THRESHOLD"}
        banned = ("route_threshold", "action_threshold", "precedence", "priority_order",
                  "interrupt_threshold", "suppress_threshold", "defer_threshold")
        for path in ANALYSIS_SOURCES:
            for name in identifiers(path) - allowed:
                for term in banned:
                    self.assertNotIn(term, name.lower(), f"{path.name}:{name}")

    def test_no_ranking_similarity_or_calibration(self):
        banned = ("ranking", "relevance_score", "similarity", "jaccard", "tfidf", "calibrat",
                  "evidence_rank", "reason_template")
        for path in ANALYSIS_SOURCES:
            for name in identifiers(path):
                for term in banned:
                    self.assertNotIn(term, name.lower(), f"{path.name}:{name}")


class ResultShapeTests(unittest.TestCase):
    def test_classification_result_has_no_decision_fields(self):
        names = {f.name for f in dataclasses.fields(ClassificationResult)}
        properties = {n for n in dir(ClassificationResult)
                      if not n.startswith("_") and isinstance(getattr(ClassificationResult, n, None), property)}
        for name in names | properties:
            for term in ("action", "route", "confidence", "rationale", "reason", "score", "rank", "evidence"):
                self.assertNotIn(term, name.lower(), name)

    def test_decision_features_has_no_decision_fields(self):
        names = {f.name for f in dataclasses.fields(DecisionFeatures)}
        properties = {n for n in dir(DecisionFeatures)
                      if not n.startswith("_") and isinstance(getattr(DecisionFeatures, n, None), property)}
        for name in names | properties:
            for term in ("action", "route", "confidence", "rationale", "reason_text",
                         "score", "ranking", "evidence_rank"):
                self.assertNotIn(term, name.lower(), name)

    def test_contributors_carry_no_content(self):
        for model in (ClassificationContributor, FeatureContributor, Feature):
            for field in dataclasses.fields(model):
                for term in ("text", "content", "payload", "url", "snippet", "excerpt", "match", "span"):
                    self.assertNotIn(term, field.name.lower(), f"{model.__name__}.{field.name}")

    def test_all_result_objects_are_frozen(self):
        for model in (ClassificationResult, ClassificationContributor,
                      DecisionFeatures, Feature, FeatureContributor):
            self.assertTrue(model.__dataclass_params__.frozen, model.__name__)

    def test_classification_and_routing_vocabularies_are_disjoint(self):
        from message_router.domain import RoutingAction

        self.assertFalse({c.value for c in MessageClass} & {a.value for a in RoutingAction})


class NoNetworkOrModelTests(unittest.TestCase):
    BANNED = ("urllib.request", "http.client", "socket", "requests", "httpx", "aiohttp",
              "openai", "anthropic", "transformers", "torch", "sklearn", "numpy",
              "cv2", "PIL", "pytesseract", "speech_recognition", "subprocess", "ssl", "dns")

    def test_no_network_model_or_engine_imports(self):
        for path in ANALYSIS_SOURCES:
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
            "import message_router.analysis\n"
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
        import re

        banned = re.compile(
            r"(?i)\b(notify|digest|mute[ds]?)\b|"
            r"\b(?:msg|hist|sample_msg|u|img|vn|group|business)_[0-9]{2,}\b|"
            r"solved[ _-]?(?:example|label)|hackerrank|\borchestrate\b"
        )
        for path in ANALYSIS_SOURCES:
            matches = banned.findall(path.read_text(encoding="utf-8"))
            self.assertEqual(matches, [], f"{path.name}: {matches}")


class SafetyIsolationTests(unittest.TestCase):
    def test_analysis_never_constructs_or_alters_safety_findings(self):
        """Risk is projected. Nothing here builds a finding or a severity."""
        for path in ANALYSIS_SOURCES:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("SafetyFinding(", source, path.name)
            self.assertNotIn("SafetyAssessment(", source, path.name)
            self.assertNotIn("Severity.SEVERE,", source, path.name)


if __name__ == "__main__":
    unittest.main()
