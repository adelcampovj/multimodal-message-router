"""Tests that keep the documentation true.

These do not test the router. They check that what the README and the files in
``docs/`` say about it can be verified: links resolve, diagrams are well formed
and show the stages in the order the code calls them, the tables and JSON
excerpts match live output, every documented command runs, and the claims that
are easiest to get wrong (counts, stale wording, what the media examples are)
are pinned.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
README = REPO_ROOT / "README.md"
DOC_FILES = [README, *sorted((REPO_ROOT / "docs").glob("*.md"))]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

FENCE_RE = re.compile(r"^```([\w-]*)[ \t]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _fences(text: str) -> list[tuple[str, str, int]]:
    """(language, body, start offset) for every fenced block."""
    return [(m.group(1), m.group(2), m.start()) for m in FENCE_RE.finditer(text)]


def _without_fences(text: str) -> str:
    return FENCE_RE.sub("", text)


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped."""
    cleaned = re.sub(r"[`*_]", "", heading).strip().lower()
    cleaned = re.sub(r"[^\w\- ]", "", cleaned)
    return cleaned.replace(" ", "-")


def _is_subset(expected, actual) -> bool:
    """Every key and value in ``expected`` is present, equal, in ``actual``."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _is_subset(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(expected) == len(actual)
                and all(_is_subset(e, a) for e, a in zip(expected, actual)))
    return expected == actual


def _runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_media_examples", REPO_ROOT / "examples" / "run_media_examples.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decision(example: str, *, runner: bool) -> dict:
    from message_router.io import load_request_file
    from message_router.media import MediaCoordinator
    from message_router.pipeline import decision_to_dict, route_request

    loaded = load_request_file(REPO_ROOT / "examples" / f"{example}.json")
    if not runner:
        return decision_to_dict(route_request(loaded))
    module = _runner_module()
    coordinator = MediaCoordinator(ocr=module.SyntheticOcrFixture(), asr=module.SyntheticAsrFixture())
    return decision_to_dict(route_request(loaded, coordinator=coordinator))


MEDIA_EXAMPLES = {"ocr_example", "asr_example"}


class LinkTests(unittest.TestCase):
    def test_relative_links_and_anchors_resolve(self):
        checked = 0
        for doc in DOC_FILES:
            for target in LINK_RE.findall(_without_fences(_text(doc))):
                if re.match(r"^[a-z][a-z0-9+.-]*:", target):
                    continue  # an absolute URL; nothing local to resolve
                path_part, _, anchor = target.partition("#")
                resolved = doc if not path_part else (doc.parent / path_part).resolve()
                self.assertTrue(resolved.exists(), f"{doc.name}: broken link {target}")
                if anchor and resolved.suffix == ".md":
                    slugs = {_slug(h) for _, h in HEADING_RE.findall(_text(resolved))}
                    self.assertIn(anchor, slugs, f"{doc.name}: no heading for #{anchor}")
                checked += 1
        self.assertGreater(checked, 5, "expected the documents to link to each other")

    def test_the_policy_reference_is_not_called_evaluation(self):
        self.assertTrue((REPO_ROOT / "docs" / "routing-policy.md").exists())
        self.assertFalse((REPO_ROOT / "docs" / "evaluation.md").exists())
        for doc in DOC_FILES:
            self.assertNotIn("evaluation.md", _text(doc), doc.name)


class MermaidTests(unittest.TestCase):
    ARROW_RE = re.compile(r"-\.->|-->|---")

    def blocks(self):
        found = []
        for doc in DOC_FILES:
            for language, body, _ in _fences(_text(doc)):
                if language == "mermaid":
                    found.append((doc.name, body))
        return found

    @staticmethod
    def _analyse(body: str):
        """(defined ids, referenced ids, edges) for one flowchart block."""
        defined, referenced, edges, subgraphs = set(), set(), set(), set()
        depth = 0
        for raw in body.splitlines():
            line = raw.strip()
            if not line or line.startswith("%%"):
                continue
            if line.startswith("flowchart "):
                continue
            if line.startswith("subgraph "):
                subgraphs.add(re.match(r"subgraph\s+(\w+)", line).group(1))
                depth += 1
                continue
            if line == "end":
                depth -= 1
                continue
            stripped = re.sub(r'"[^"]*"', '""', line)
            stripped = re.sub(r"\|[^|]*\|", "", stripped)
            for opener, closer in (("[", "]"), ("(", ")"), ("{", "}")):
                if stripped.count(opener) != stripped.count(closer):
                    raise AssertionError(f"unbalanced {opener}{closer} in: {line}")
            if line.count('"') % 2:
                raise AssertionError(f"odd number of quotes in: {line}")
            ids = []
            for segment in MermaidTests.ARROW_RE.split(stripped):
                match = re.match(r"\s*([A-Za-z_]\w*)\s*([\[({]?)", segment)
                if not match:
                    raise AssertionError(f"cannot read a node from: {line}")
                node, opener = match.group(1), match.group(2)
                referenced.add(node)
                ids.append(node)
                if opener:
                    defined.add(node)
            edges.update(zip(ids, ids[1:]))
        if depth != 0:
            raise AssertionError("subgraph/end are unbalanced")
        return defined, referenced, edges, subgraphs

    def test_each_diagram_is_well_formed(self):
        blocks = self.blocks()
        self.assertGreaterEqual(len(blocks), 3, "expected diagrams in the README and docs")
        for name, body in blocks:
            with self.subTest(document=name):
                self.assertRegex(body.lstrip(), r"^flowchart (TD|LR|TB|RL)\b")
                defined, referenced, edges, subgraphs = self._analyse(body)
                self.assertTrue(edges)
                self.assertFalse(referenced - defined - subgraphs,
                                 f"nodes used but never defined: {referenced - defined}")
                self.assertNotIn("end", defined | referenced)

    def test_the_pipeline_diagram_orders_stages_as_the_code_calls_them(self):
        """Derive the order from pipeline.py itself rather than restating it."""
        tree = ast.parse(_text(SRC_DIR / "message_router" / "pipeline.py"))
        func = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "route_request")
        called = []
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                name = (node.func.id if isinstance(node.func, ast.Name)
                        else node.func.attr if isinstance(node.func, ast.Attribute) else None)
                if name in {"analyze", "analyze_message", "classify", "extract_features", "route"}:
                    called.append((node.lineno, name))
        order = [name for _, name in sorted(called)]
        self.assertEqual(order, ["analyze", "analyze_message", "classify", "extract_features", "route"])

        node_for = {"analyze": "MEDIA", "analyze_message": "SAFETY", "classify": "CLASSIFY",
                    "extract_features": "FEATURES", "route": "POLICY"}
        architecture = next(body for doc, body in self.blocks() if doc == "architecture.md")
        _, _, edges, _ = self._analyse(architecture)
        graph = {}
        for a, b in edges:
            graph.setdefault(a, set()).add(b)

        def reaches(start, goal, seen=()):
            return start == goal or any(
                reaches(nxt, goal, (*seen, start)) for nxt in graph.get(start, ()) if nxt not in seen)

        for first, second in zip(order, order[1:]):
            self.assertTrue(reaches(node_for[first], node_for[second]),
                            f"diagram does not show {first} before {second}")
        self.assertTrue(reaches("DOC", "LOAD") and reaches("LOAD", "MEDIA"))
        self.assertTrue(reaches("POLICY", "OUT"))
        self.assertFalse(reaches("SAFETY", "MEDIA"), "safety must not feed the media readings")
        self.assertFalse(reaches("POLICY", "SAFETY"), "the policy must not feed back into analysis")

    def test_the_media_diagram_does_not_show_the_route_command_decoding_media(self):
        body = next(b for d, b in self.blocks() if d == "architecture.md" and "SG_A" in b)
        self.assertIn("no backends", body)
        self.assertIn("SYNTHETIC", body)
        self.assertIn("NOT included", body)
        self.assertNotRegex(body, r"(?i)route command[^\"]*\b(decodes|transcribes|recognises)\b")


class DemoMatchesTheRouterTests(unittest.TestCase):
    ROW_RE = re.compile(r"\|[^|\n]*`examples/(\w+)\.json`[^|\n]*\|\s*`(defer|interrupt|suppress)`\s*\|"
                        r"\s*`([\w.]+)`\s*\|\s*`(\w+)`\s*\|")

    def test_the_example_tables_match_live_decisions(self):
        seen = set()
        for doc in (README, REPO_ROOT / "docs" / "demo.md"):
            for example, action, policy, classification in self.ROW_RE.findall(_text(doc)):
                with self.subTest(document=doc.name, example=example):
                    actual = _decision(example, runner=example in MEDIA_EXAMPLES)
                    self.assertEqual(
                        (action, policy, classification),
                        (actual["action"], actual["policy_id"], actual["classification"]["selected"]))
                seen.add((doc.name, example))
        for doc_name in ("README.md", "demo.md"):
            self.assertEqual(
                {e for d, e in seen if d == doc_name},
                {"benign", "urgent", "credential_request", "suspicious_payment",
                 "ocr_example", "asr_example"}, doc_name)

    def test_every_json_excerpt_in_the_demo_is_a_true_subset_of_the_real_output(self):
        text = _text(REPO_ROOT / "docs" / "demo.md")
        fences = _fences(text)
        checked = 0
        for index, (language, body, _) in enumerate(fences):
            if language != "json" or '"action"' not in body or '"format"' in body:
                continue
            command = next(b for l, b, _ in reversed(fences[:index]) if l == "bash")
            example = re.findall(r"examples/(\w+)\.json", command)[-1]
            via_runner = "run_media_examples" in command.strip().splitlines()[-1]
            actual = _decision(example, runner=via_runner)
            with self.subTest(example=example):
                self.assertTrue(_is_subset(json.loads(body), actual), f"excerpt for {example} drifted")
            checked += 1
        self.assertEqual(checked, 6)

    def test_the_standalone_route_command_reads_no_media(self):
        for example in sorted(MEDIA_EXAMPLES):
            with self.subTest(example=example):
                decision = _decision(example, runner=False)
                self.assertTrue(decision["analysis_incomplete"])
                self.assertEqual(decision["safety"]["findings"], [])
                self.assertEqual(decision["action"], "defer")

    def test_the_documented_history_and_preference_behaviour_is_true(self):
        from message_router.io import load_request
        from message_router.pipeline import decision_to_dict, route_request

        base = json.loads(_text(REPO_ROOT / "examples" / "benign.json"))

        def run(**request_changes):
            doc = json.loads(json.dumps(base))
            doc["request"].update(request_changes)
            return decision_to_dict(route_request(load_request(json.dumps(doc).encode("utf-8"))))

        history = [{"id": "rec-1", "occurred_at": "2031-03-09T09:01:00+00:00",
                    "conversation_id": "conv-demo", "opened": True, "replied": True,
                    "dismissed": False}]
        self.assertEqual(run(), run(history=history), "history must not influence the decision")
        silenced = run(recipient={"id": "rcp-demo",
                                  "preferences": {"silenced_conversations": ["conv-demo"]}})
        self.assertNotEqual(run()["contributors"], silenced["contributors"],
                            "preferences are documented as consumed")


class DocumentedCommandsTests(unittest.TestCase):
    RUNNABLE = ("python -m message_router", "python examples/run_media_examples.py")

    def commands(self):
        found = []
        for doc in (README, REPO_ROOT / "docs" / "demo.md"):
            for language, body, _ in _fences(_text(doc)):
                if language not in {"bash", "powershell"}:
                    continue
                for line in body.splitlines():
                    line = line.strip()
                    if line.startswith("PYTHONPATH=src "):
                        line = line[len("PYTHONPATH=src "):]
                    if line.startswith(self.RUNNABLE) and line not in found:
                        found.append(line)
        return found

    def test_every_documented_command_runs_successfully(self):
        commands = self.commands()
        self.assertGreaterEqual(len(commands), 9, "expected the quickstart and all six demos")
        env = {**os.environ, "PYTHONPATH": str(SRC_DIR), "PYTHONDONTWRITEBYTECODE": "1"}
        for command in commands:
            with self.subTest(command=command):
                parts = command.split()
                result = subprocess.run([sys.executable, *parts[1:]], capture_output=True, text=True,
                                        timeout=60, env=env, cwd=str(REPO_ROOT))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.strip())

    def test_all_six_examples_have_a_documented_command(self):
        joined = "\n".join(self.commands())
        for example in ("benign", "urgent", "credential_request", "suspicious_payment",
                        "ocr_example", "asr_example"):
            self.assertIn(f"examples/{example}.json", joined)

    def test_the_test_and_audit_commands_are_documented_with_the_bytecode_variable(self):
        text = _text(README)
        self.assertIn("python -m unittest discover -s tests", text)
        self.assertIn("python scripts/audit_public_tree.py", text)
        self.assertIn("PYTHONDONTWRITEBYTECODE", text)


class ClaimsTests(unittest.TestCase):
    def test_counts_in_the_prose_match_the_code(self):
        from message_router.analysis import FeatureFamily, MessageClass
        from message_router.routing import POLICY_ORDER

        expected = {"ordered policies": len(POLICY_ORDER), "message classes": len(MessageClass),
                    "feature families": len(FeatureFamily)}
        for doc in (README, REPO_ROOT / "docs" / "architecture.md"):
            for count, noun in re.findall(r"\b(\d+) (ordered policies|message classes|feature families)",
                                          _text(doc)):
                self.assertEqual(int(count), expected[noun], f"{doc.name}: {count} {noun}")
        readme = _text(README)
        for noun in expected:
            self.assertRegex(readme, rf"\b\d+ {noun}")

    def test_the_actions_are_the_three_real_ones(self):
        from message_router.domain import RoutingAction

        self.assertEqual({a.value for a in RoutingAction}, {"interrupt", "defer", "suppress"})
        readme = _text(README)
        for action in ("interrupt", "defer", "suppress"):
            self.assertIn(f"`{action}`", readme)

    def test_no_stale_or_inflated_wording_survives(self):
        stale = ("scaffold", "not implemented yet", "under construction", "reserved for a later",
                 "has not been implemented", "currently a scaffold")
        files = [*DOC_FILES, REPO_ROOT / "pyproject.toml",
                 SRC_DIR / "message_router" / "cli.py", SRC_DIR / "message_router" / "__init__.py"]
        for path in files:
            lowered = _text(path).lower()
            for phrase in stale:
                self.assertNotIn(phrase, lowered, f"{path.name}: stale phrase {phrase!r}")

    def test_no_unfilled_placeholder_is_left_in_any_document(self):
        for doc in DOC_FILES:
            self.assertNotIn("@@", _text(doc), f"{doc.name} still has a placeholder")

    def test_no_numeric_effectiveness_claim_is_made(self):
        readme = _text(README)
        self.assertNotRegex(readme, r"\d\s*%", "the README must not quote percentages")
        self.assertNotRegex(readme.lower(), r"\b(accuracy|precision|recall|f1)\b\s*(of|:|=|is|was)\s*\d")

    def test_the_media_examples_are_described_as_synthetic_in_both_places(self):
        for path in (README, REPO_ROOT / "docs" / "demo.md"):
            lowered = " ".join(_text(path).lower().split())
            self.assertIn("pre-extracted synthetic", lowered, path.name)
            self.assertRegex(lowered, r"no (ocr|image|speech)[^.]*(engine|recognition)", path.name)

    def test_the_origin_is_described_modestly(self):
        lowered = " ".join(_text(README).lower().split())
        self.assertIn("not an official submission", lowered)
        self.assertNotRegex(lowered, r"\b(winner|winning|award[- ]winning|leaderboard|first place)\b")

    def test_the_readme_has_its_sections_in_order(self):
        headings = [h.strip() for level, h in HEADING_RE.findall(_text(README)) if len(level) == 2]
        expected = ["The problem", "What it does", "How it works", "Engineering design highlights",
                    "Quickstart", "Demonstrations", "Testing and CI", "Known limitations",
                    "Future improvements"]
        positions = [headings.index(h) for h in expected]
        self.assertEqual(positions, sorted(positions))

    def test_the_limitations_the_project_actually_has_are_disclosed(self):
        lowered = " ".join(_text(README).lower().split())
        for fragment in ("history", "preferences", "no ocr", "external", "password policy",
                         "plural", "confidence", "natural-language", "real-world"):
            self.assertIn(fragment, lowered, fragment)


class PackageMetadataTests(unittest.TestCase):
    """Read with a regular expression because ``tomllib`` needs Python 3.11 and
    the project supports 3.10. Only the few simple keys checked here are read."""

    @staticmethod
    def _project_table() -> str:
        text = _text(REPO_ROOT / "pyproject.toml")
        match = re.search(r"^\[project\]\n(.*?)(?=^\[)", text, re.DOTALL | re.MULTILINE)
        assert match, "no [project] table"
        return match.group(1)

    def _value(self, key: str) -> str:
        match = re.search(rf"^{re.escape(key)}\s*=\s*(.+)$", self._project_table(), re.MULTILINE)
        self.assertIsNotNone(match, f"pyproject has no {key}")
        return match.group(1).strip()

    def test_metadata_describes_the_working_project_and_keeps_its_identity(self):
        self.assertEqual(self._value("name"), '"multimodal-message-router"')
        self.assertEqual(self._value("version"), '"0.0.0"')
        self.assertEqual(self._value("requires-python"), '">=3.10"')
        self.assertEqual(self._value("dependencies"), "[]")
        description = self._value("description").lower()
        self.assertNotIn("scaffold", description)
        for word in ("interrupt", "defer", "suppress"):
            self.assertIn(word, description)
        whole = _text(REPO_ROOT / "pyproject.toml")
        self.assertNotIn("[project.urls]", whole, "no repository URL is invented")
        self.assertNotRegex(whole, r"(?m)^authors\s*=", "no author identity is invented")

    def test_the_test_suite_itself_needs_nothing_newer_than_the_declared_python(self):
        """``tomllib`` (3.11+) is the kind of import that would break 3.10."""
        for path in sorted((REPO_ROOT / "tests").glob("*.py")):
            for node in ast.walk(ast.parse(_text(path))):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                self.assertNotIn("tomllib", [n.split(".")[0] for n in names], path.name)


if __name__ == "__main__":
    unittest.main()
