"""Structural guarantees about the safety layer.

These do not test behaviour. They assert that the safety package stays
inside its remit: it reports risk, and it does not decide delivery, rank
evidence, import a model, or reach the network.
"""

from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SAFETY_DIR = SRC_DIR / "message_router" / "safety"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from message_router.safety import SafetyAssessment, SafetyFinding, Severity  # noqa: E402

SAFETY_SOURCES = sorted(SAFETY_DIR.glob("*.py"))


def all_identifiers(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


class AssessmentBoundaryTests(unittest.TestCase):
    def test_assessment_has_no_decision_fields(self):
        field_names = {f.name for f in dataclasses.fields(SafetyAssessment)}
        properties = {
            name for name in dir(SafetyAssessment) if not name.startswith("_")
        } - field_names
        surface = field_names | properties
        banned = (
            "action", "routing", "category_decision", "message_category", "rationale",
            "reason", "confidence", "score", "rank", "relevance", "preference",
            "engagement", "utility", "urgency_score", "trust_score",
        )
        for name in surface:
            for term in banned:
                self.assertNotIn(term, name.lower(), f"{name} looks like a decision field")

    def test_finding_has_no_decision_fields_and_no_content(self):
        field_names = {f.name for f in dataclasses.fields(SafetyFinding)}
        banned = ("action", "confidence", "rationale", "reason", "score", "rank",
                  "text", "content", "payload", "url", "snippet", "excerpt", "span", "match")
        for name in field_names:
            for term in banned:
                self.assertNotIn(term, name.lower(), name)

    def test_hard_veto_is_derived_not_settable(self):
        """The veto flag is computed from findings, so nothing can assert it
        independently of the evidence for it."""
        self.assertNotIn("hard_veto_required", {f.name for f in dataclasses.fields(SafetyAssessment)})
        self.assertIsInstance(SafetyAssessment.hard_veto_required, property)
        empty = SafetyAssessment()
        self.assertFalse(empty.hard_veto_required)
        self.assertIsNone(empty.highest_severity)

    def test_only_severe_requires_a_veto(self):
        self.assertTrue(Severity.SEVERE.requires_hard_veto)
        self.assertFalse(Severity.SUSPICIOUS.requires_hard_veto)
        self.assertFalse(Severity.WEAK.requires_hard_veto)

    def test_assessment_and_finding_are_immutable(self):
        finding = SafetyFinding.__dataclass_fields__
        self.assertTrue(SafetyFinding.__dataclass_params__.frozen)
        self.assertTrue(SafetyAssessment.__dataclass_params__.frozen)
        self.assertTrue(finding)


class NoRoutingOrRankingTests(unittest.TestCase):
    def test_safety_never_constructs_a_routing_decision(self):
        banned_constructions = (
            "RoutingDecision(", "Confidence(", "Rationale(", "RoutingAction.", "MessageCategory.",
        )
        for path in SAFETY_SOURCES:
            source = path.read_text(encoding="utf-8")
            for banned in banned_constructions:
                self.assertNotIn(banned, source, f"{path.name} references {banned}")

    def test_safety_does_not_import_routing_or_decision_modules(self):
        for path in SAFETY_SOURCES:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertNotIn("decision", node.module)
                    self.assertNotIn("routing", node.module)

    def test_safety_defines_no_ranking_or_calibration_identifiers(self):
        """Bans *evidence* ranking and confidence calibration.

        A severity ordinal (``Severity.rank``) is deliberately not banned:
        comparing "severe" against "weak" needs an ordering, and that is a
        different thing from ranking evidence by relevance.
        """
        banned = ("ranking", "rank_evidence", "evidence_rank", "relevance",
                  "similarity", "calibrat", "tfidf", "jaccard", "utility_score")
        offenders = []
        for path in SAFETY_SOURCES:
            for name in all_identifiers(path):
                if any(term in name.lower() for term in banned):
                    offenders.append(f"{path.name}:{name}")
        self.assertEqual(offenders, [])


class NoNetworkOrModelTests(unittest.TestCase):
    BANNED_MODULES = (
        "urllib.request", "http.client", "socket", "requests", "httpx", "aiohttp",
        "openai", "anthropic", "transformers", "torch", "sklearn", "numpy",
        "cv2", "PIL", "pytesseract", "speech_recognition", "dns", "subprocess", "ssl",
    )

    def test_no_network_or_model_or_engine_imports(self):
        for path in SAFETY_SOURCES:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    for banned in self.BANNED_MODULES:
                        self.assertFalse(
                            module == banned or module.startswith(banned + "."),
                            f"{path.name} imports {module}",
                        )

    def test_only_expected_third_party_free_imports_are_used(self):
        allowed_roots = {
            "__future__", "dataclasses", "enum", "re", "unicodedata", "ipaddress",
            "urllib", "typing", "collections", "datetime", "math",
        }
        for path in SAFETY_SOURCES:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertIn(alias.name.split(".")[0], allowed_roots, path.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:  # relative import inside this package
                        continue
                    self.assertIn(node.module.split(".")[0], allowed_roots, path.name)

    def test_safety_imports_in_a_clean_subprocess_with_engines_blocked(self):
        script = (
            "import sys\n"
            "blocked = {'cv2','PIL','numpy','torch','requests','httpx','openai','anthropic'}\n"
            "class Blocker:\n"
            "    def find_module(self, name, path=None):\n"
            "        if name.split('.')[0] in blocked:\n"
            "            raise ImportError('blocked: ' + name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "import message_router.safety\n"
            "print('IMPORT_OK')\n"
        )
        import os

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(SRC_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("IMPORT_OK", result.stdout)


class NoOrganizerVocabularyTests(unittest.TestCase):
    def test_no_frozen_action_aliases_or_identifiers(self):
        import re

        banned = re.compile(
            r"(?i)\b(notify|digest|mute[ds]?)\b|"
            r"\b(?:msg|hist|sample_msg|u|img|vn|group|business)_[0-9]{2,}\b|"
            r"solved[ _-]?(?:example|label)|hackerrank|\borchestrate\b"
        )
        for path in SAFETY_SOURCES:
            matches = banned.findall(path.read_text(encoding="utf-8"))
            self.assertEqual(matches, [], f"{path.name}: {matches}")


if __name__ == "__main__":
    unittest.main()
