"""Tests for the message_router CLI scaffold."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

FORBIDDEN_TERMS = (
    "HackerRank",
    "Orchestrate",
    "organizer",
    "challenge",
    "AI Judge",
    "submission",
    "leaderboard",
    "30/30",
)


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SRC_DIR)}
    return subprocess.run(
        [sys.executable, "-m", "message_router", *args],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        cwd=str(REPO_ROOT),
    )


class PackageImportTests(unittest.TestCase):
    def test_package_imports(self):
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        import message_router

        self.assertTrue(hasattr(message_router, "__version__"))


class CliInvocationTests(unittest.TestCase):
    def test_help_exits_zero(self):
        result = _run(["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("message_router", result.stdout)

    def test_no_arguments_exits_zero(self):
        result = _run([])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_under_construction_message_present(self):
        result = _run([])
        lowered = result.stdout.lower()
        self.assertIn("under construction", lowered)
        self.assertIn("scaffold", lowered)

    def test_no_forbidden_language_in_help_or_default_output(self):
        combined = _run(["--help"]).stdout + "\n" + _run([]).stdout
        lowered = combined.lower()
        for term in FORBIDDEN_TERMS:
            self.assertNotIn(term.lower(), lowered, f"forbidden term leaked: {term}")


class MainDelegationTests(unittest.TestCase):
    def test_main_module_delegates_to_cli_main(self):
        main_path = SRC_DIR / "message_router" / "__main__.py"
        source = main_path.read_text(encoding="utf-8")
        self.assertIn("from .cli import main", source)

        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "main"
        ]
        self.assertTrue(calls, "__main__.py must call main()")


if __name__ == "__main__":
    unittest.main()
