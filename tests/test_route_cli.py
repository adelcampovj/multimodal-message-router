"""Tests for the ``route`` command and the synthetic examples.

The CLI is a thin caller of stages that already exist, so these tests check
two things: that it faithfully reports what those stages decide (compared
against the stages called directly, not against a hard-coded copy), and that
it is honest about what it does not do -- it reads no images, transcribes no
audio, and prints nothing from the request.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES = REPO_ROOT / "examples"
RUNNER = EXAMPLES / "run_media_examples.py"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Same discipline as tests/test_cli.py: assembled from fragments so this
# denylist is not itself the one place the literals appear.
FORBIDDEN_TERMS = (
    "Hacker" + "Rank",
    "Orche" + "strate",
    "organ" + "izer",
    "chall" + "enge",
    "AI " + "Judge",
    "submis" + "sion",
    "leader" + "board",
)

TEXT_EXAMPLES = {
    "benign": ("defer", "route.utility.safe_defer"),
    "urgent": ("interrupt", "route.urgency.safe_interrupt"),
    "credential_request": ("suppress", "route.safety.hard_veto"),
    "suspicious_payment": ("suppress", "route.safety.hard_veto"),
}
MEDIA_EXAMPLES = ("ocr_example", "asr_example")

#: The only keys a decision may carry. Anything else would be new surface.
DECISION_KEYS = {
    "action", "analysis_incomplete", "classification", "contributors",
    "diagnostics", "hard_veto_applied", "policy_id", "safety",
}


def _env() -> dict:
    return {**os.environ, "PYTHONPATH": str(SRC_DIR), "PYTHONDONTWRITEBYTECODE": "1"}


def _route(path: Path | str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "message_router", "route", "--input", str(path)],
        capture_output=True, text=True, timeout=60, env=_env(), cwd=str(REPO_ROOT),
    )


def _runner(path: Path | str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), "--input", str(path)],
        capture_output=True, text=True, timeout=60, env=_env(), cwd=str(REPO_ROOT),
    )


def _example(name: str) -> Path:
    return EXAMPLES / f"{name}.json"


def _stages_directly(path: Path):
    """The existing stages, called by hand -- the independent oracle."""
    from message_router.analysis import classify, extract_features
    from message_router.io import load_request_file
    from message_router.routing import route
    from message_router.safety import analyze_message

    loaded = load_request_file(path)
    message = loaded.message
    assessment = analyze_message(message)
    texts = ((message.body.value, message.body.provenance),)
    classification = classify(texts, assessment)
    features = extract_features(message, assessment, classification, preferences=loaded.preferences)
    return classification, route(assessment, classification, features)


class RouteExamplesTests(unittest.TestCase):
    """Each text example reaches the action it is documented to reach."""

    def test_each_text_example_reaches_its_documented_action(self):
        for name, (action, policy) in TEXT_EXAMPLES.items():
            with self.subTest(example=name):
                result = _route(_example(name))
                self.assertEqual(result.returncode, 0, result.stderr)
                decision = json.loads(result.stdout)
                self.assertEqual(decision["action"], action)
                self.assertEqual(decision["policy_id"], policy)

    def test_output_is_exactly_one_json_document_with_known_keys(self):
        for name in TEXT_EXAMPLES:
            with self.subTest(example=name):
                result = _route(_example(name))
                decision = json.loads(result.stdout)  # raises if not one document
                self.assertEqual(set(decision), DECISION_KEYS)
                self.assertTrue(result.stdout.endswith("}\n"))
                self.assertEqual(result.stderr, "")

    def test_routing_fields_match_the_existing_stages_exactly(self):
        for name in TEXT_EXAMPLES:
            with self.subTest(example=name):
                classification, routed = _stages_directly(_example(name))
                decision = json.loads(_route(_example(name)).stdout)
                expected = dict(routed.serialize())
                self.assertEqual(decision["action"], expected["action"])
                self.assertEqual(decision["policy_id"], expected["policy_id"])
                self.assertEqual(decision["hard_veto_applied"], expected["hard_veto_applied"])
                self.assertEqual(decision["analysis_incomplete"], expected["analysis_incomplete"])
                self.assertEqual(decision["diagnostics"], list(expected["diagnostics"]))
                self.assertEqual(decision["classification"]["selected"], classification.selected.value)
                self.assertEqual(
                    [(c["family"], c["level"], c["effect"], tuple(c["rule_ids"]))
                     for c in decision["contributors"]],
                    [tuple(c) for c in expected["contributors"]],
                )

    def test_repeated_routing_is_byte_identical(self):
        for name in TEXT_EXAMPLES:
            with self.subTest(example=name):
                self.assertEqual(_route(_example(name)).stdout, _route(_example(name)).stdout)

    def test_credential_and_payment_findings_are_attributed_to_authored_text(self):
        for name, rule in (("credential_request", "CRED_SECRET_DISCLOSURE"),
                           ("suspicious_payment", "PAY_URGENT_DEMAND")):
            with self.subTest(example=name):
                findings = json.loads(_route(_example(name)).stdout)["safety"]["findings"]
                self.assertIn(rule, [f["rule_id"] for f in findings])
                self.assertEqual({f["provenance"] for f in findings}, {"authored_text"})

    def test_output_carries_nothing_from_the_request(self):
        for name in TEXT_EXAMPLES:
            with self.subTest(example=name):
                document = json.loads(_example(name).read_text(encoding="utf-8"))
                body = document["request"]["message"]["body"]
                out = _route(_example(name)).stdout
                self.assertNotIn(body, out)
                for fragment in ("203.0.113.9", "one time code", "evacuate", "helpdesk", "parcel"):
                    self.assertNotIn(fragment, out)
                for identifier in ("rcp-demo", "conv-demo", f"note-{name.split('_')[0]}"):
                    self.assertNotIn(identifier, out)

    def test_every_example_is_a_valid_request_document(self):
        for name in (*TEXT_EXAMPLES, *MEDIA_EXAMPLES):
            with self.subTest(example=name):
                result = subprocess.run(
                    [sys.executable, "-m", "message_router", "validate", str(_example(name))],
                    capture_output=True, text=True, timeout=60, env=_env(), cwd=str(REPO_ROOT),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("result=VALID", result.stdout)

    def test_payment_example_uses_a_reserved_documentation_address(self):
        body = json.loads(_example("suspicious_payment").read_text(encoding="utf-8"))
        text = body["request"]["message"]["body"]
        self.assertIn("203.0.113.", text)  # RFC 5737 TEST-NET-3


class RouteErrorTests(unittest.TestCase):
    """Failures are nonzero, useful, and silent about the request."""

    CANARY = "zqxjv-body-canary-4471"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="route_cli_"))
        self.addCleanup(_rmtree, self.tmp)

    def _write(self, name: str, content) -> Path:
        path = self.tmp / name
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        return path

    def _valid(self) -> dict:
        document = json.loads(_example("benign").read_text(encoding="utf-8"))
        document["request"]["message"]["body"] = self.CANARY
        return document

    def assert_clean_failure(self, result: subprocess.CompletedProcess):
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "", "a failed route must print no decision")
        self.assertIn("error=", result.stderr)
        self.assertNotIn(self.CANARY, result.stdout + result.stderr)

    def test_missing_file_is_a_nonzero_clean_failure(self):
        self.assert_clean_failure(_route(self.tmp / "does-not-exist.json"))

    def test_malformed_json_is_a_nonzero_clean_failure(self):
        self.assert_clean_failure(_route(self._write("bad.json", '{"request": ' + self.CANARY)))

    def test_unknown_field_is_a_nonzero_clean_failure(self):
        document = self._valid()
        document["request"]["message"]["verdict"] = self.CANARY
        self.assert_clean_failure(_route(self._write("unknown.json", document)))

    def test_wrong_format_is_a_nonzero_clean_failure(self):
        document = self._valid()
        document["format"] = self.CANARY
        self.assert_clean_failure(_route(self._write("format.json", document)))

    def test_missing_required_field_is_a_nonzero_clean_failure(self):
        document = self._valid()
        del document["request"]["recipient"]
        self.assert_clean_failure(_route(self._write("missing.json", document)))

    def test_routing_failure_reports_the_type_but_not_the_message(self):
        from message_router import cli

        stderr, stdout = io.StringIO(), io.StringIO()
        with mock.patch("message_router.pipeline.route_request",
                        side_effect=RuntimeError(self.CANARY)), \
                contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
            status = cli.main(["route", "--input", str(_example("benign"))])
        self.assertEqual(status, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("RuntimeError", stderr.getvalue())
        self.assertNotIn(self.CANARY, stderr.getvalue())

    def test_input_is_required(self):
        result = subprocess.run(
            [sys.executable, "-m", "message_router", "route"],
            capture_output=True, text=True, timeout=60, env=_env(), cwd=str(REPO_ROOT),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")


class RouteDoesNotDisturbExistingCommandsTests(unittest.TestCase):
    def test_validate_still_loads_only(self):
        result = subprocess.run(
            [sys.executable, "-m", "message_router", "validate", str(_example("credential_request"))],
            capture_output=True, text=True, timeout=60, env=_env(), cwd=str(REPO_ROOT),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("result=VALID", result.stdout)
        for term in ("suppress", "interrupt", "defer", "severe", "veto"):
            self.assertNotIn(term, result.stdout.lower())

    def test_route_help_is_available_and_free_of_forbidden_language(self):
        result = subprocess.run(
            [sys.executable, "-m", "message_router", "route", "--help"],
            capture_output=True, text=True, timeout=60, env=_env(), cwd=str(REPO_ROOT),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--input", result.stdout)
        lowered = result.stdout.lower()
        for term in FORBIDDEN_TERMS:
            self.assertNotIn(term.lower(), lowered)

    def test_top_level_help_lists_both_commands(self):
        result = subprocess.run(
            [sys.executable, "-m", "message_router", "--help"],
            capture_output=True, text=True, timeout=60, env=_env(), cwd=str(REPO_ROOT),
        )
        self.assertIn("validate", result.stdout)
        self.assertIn("route", result.stdout)


class MediaExampleTests(unittest.TestCase):
    """OCR and ASR are demonstrated with pre-extracted SYNTHETIC text."""

    def test_route_alone_reads_no_attachment(self):
        """Without backends the attachment is reported unread, not recognised."""
        for name in MEDIA_EXAMPLES:
            with self.subTest(example=name):
                result = _route(_example(name))
                self.assertEqual(result.returncode, 0, result.stderr)
                decision = json.loads(result.stdout)
                self.assertTrue(decision["analysis_incomplete"])
                self.assertEqual(decision["safety"]["findings"], [])
                self.assertEqual(decision["action"], "defer")

    def test_ocr_example_is_routed_with_ocr_provenance(self):
        result = _runner(_example("ocr_example"))
        self.assertEqual(result.returncode, 0, result.stderr)
        decision = json.loads(result.stdout)
        self.assertEqual(decision["action"], "suppress")
        self.assertEqual(decision["policy_id"], "route.safety.hard_veto")
        findings = decision["safety"]["findings"]
        self.assertTrue(findings)
        self.assertEqual({f["provenance"] for f in findings}, {"ocr_text"})

    def test_asr_example_is_routed_with_asr_provenance(self):
        result = _runner(_example("asr_example"))
        self.assertEqual(result.returncode, 0, result.stderr)
        decision = json.loads(result.stdout)
        self.assertEqual(decision["action"], "suppress")
        self.assertEqual(decision["policy_id"], "route.safety.hard_veto")
        findings = decision["safety"]["findings"]
        self.assertTrue(findings)
        self.assertEqual({f["provenance"] for f in findings}, {"asr_transcript"})

    def test_media_decisions_are_deterministic_and_content_free(self):
        for name in MEDIA_EXAMPLES:
            with self.subTest(example=name):
                first, second = _runner(_example(name)), _runner(_example(name))
                self.assertEqual(first.stdout, second.stdout)
                self.assertEqual(set(json.loads(first.stdout)), DECISION_KEYS)
                for fragment in ("203.0.113.9", "one time code", "helpdesk", "parcel"):
                    self.assertNotIn(fragment, first.stdout)

    def test_runner_refuses_a_document_it_has_no_synthetic_text_for(self):
        result = _runner(_example("benign"))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("error=", result.stderr)

    def test_the_documents_carry_no_extracted_text_and_no_media_is_shipped(self):
        """The format has no field for extracted text, and none is smuggled in."""
        for name in MEDIA_EXAMPLES:
            with self.subTest(example=name):
                message = json.loads(_example(name).read_text(encoding="utf-8"))["request"]["message"]
                self.assertNotIn("body", message)
                self.assertEqual(set(message["attachment"]), {"id", "kind", "source"})
        shipped = sorted(p.suffix for p in EXAMPLES.iterdir())
        self.assertEqual(set(shipped), {".json", ".py"}, "examples must contain no media files")

    def test_the_runner_describes_itself_as_synthetic_fixtures(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("PRE-EXTRACTED SYNTHETIC", source)
        self.assertIn("reads no image", source.lower())
        self.assertIn("decodes no audio", source.lower())

    def test_the_route_help_does_not_claim_recognition(self):
        result = subprocess.run(
            [sys.executable, "-m", "message_router", "route", "--help"],
            capture_output=True, text=True, timeout=60, env=_env(), cwd=str(REPO_ROOT),
        )
        flattened = " ".join(result.stdout.lower().split())
        self.assertIn("not read", flattened)
        self.assertIn("no image recognition or transcription", flattened)


class CliCarriesNoRoutingLogicTests(unittest.TestCase):
    """The replacement for the scaffold-era "CLI exposes no route command".

    What matters is not that the CLI cannot route -- ``route`` is the point of
    this stage -- but that it contains no routing logic of its own. It may
    load a document and call the shared pipeline, and nothing else.
    """

    ALLOWED_PACKAGE_IMPORTS = {"io", "pipeline"}
    STAGE_PACKAGES = {"safety", "analysis", "routing", "media", "domain", "ports", "validation"}

    def test_cli_imports_only_the_loader_and_the_pipeline(self):
        import ast

        tree = ast.parse((SRC_DIR / "message_router" / "cli.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(imported, "expected the CLI to import from the package")
        self.assertLessEqual(imported, self.ALLOWED_PACKAGE_IMPORTS, sorted(imported))
        self.assertFalse(imported & self.STAGE_PACKAGES)

    def test_the_pipeline_module_adds_no_arithmetic_or_aggregation(self):
        """The same discipline the routing package is held to."""
        import ast

        tree = ast.parse((SRC_DIR / "message_router" / "pipeline.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                self.assertNotIsInstance(node.op, (ast.Add, ast.Mult, ast.Sub, ast.Div))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, ("sum", "max", "min"))


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
