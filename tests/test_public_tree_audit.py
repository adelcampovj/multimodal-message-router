"""Tests for scripts/audit_public_tree.py.

All negative fixtures are built inside temporary directories under the
system temp directory (proven, per test, to be outside this repository) and
deleted in tearDown. No prohibited fixture content is ever committed.
"""

from __future__ import annotations

import ast
import contextlib
import getpass
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_public_tree.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_public_tree", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Must be registered before exec: dataclasses resolves its owning module
    # via sys.modules while the class body executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()


def _categories(findings):
    return {f.category for f in findings}


def _paths(findings):
    return {f.relative_path for f in findings}


class TempTreeTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="audit_public_tree_test_")).resolve()
        # Prove isolation: the fixture tree is outside this repository.
        self.assertFalse(str(self.tmp).startswith(str(REPO_ROOT)))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, relpath: str, content: str = "") -> Path:
        p = self.tmp / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p


# ---------------------------------------------------------------------------
# Clean tree.
# ---------------------------------------------------------------------------


class CleanTreeTests(TempTreeTestCase):
    def test_clean_repository_passes(self):
        self.write("README.md", "# hello\n")
        self.write("src/message_router/__init__.py", "")
        self.write("src/message_router/cli.py", "def main():\n    return 0\n")
        findings = audit.audit_tree(self.tmp)
        self.assertEqual(findings, [])


# ---------------------------------------------------------------------------
# Prohibited filenames / paths / caches.
# ---------------------------------------------------------------------------


class ProhibitedFilenameTests(TempTreeTestCase):
    def test_prohibited_filename_fails(self):
        self.write("log.txt", "irrelevant")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("prohibited_filename", _categories(findings))
        self.assertIn("log.txt", _paths(findings))

    def test_prohibited_nested_path_fails(self):
        self.write("deep/nested/path/AGENTS.md", "irrelevant")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("prohibited_filename", _categories(findings))
        self.assertIn("deep/nested/path/AGENTS.md", _paths(findings))

    def test_env_variants_fail(self):
        self.write(".env", "X=1")
        self.write(".env.local", "X=1")
        findings = audit.audit_tree(self.tmp)
        cats = _categories(findings)
        self.assertIn("prohibited_filename", cats)
        self.assertIn("env_variant_file", cats)

    def test_cache_directory_fails(self):
        self.write("src/__pycache__/mod.cpython-312.pyc", "junk")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("prohibited_directory", _categories(findings))
        self.assertIn("src/__pycache__", _paths(findings))

    def test_submission_directory_fails(self):
        self.write("submission/code.zip", "junk")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("prohibited_directory", _categories(findings))

    def test_ocr_asr_cache_artifact_fails(self):
        self.write("ocr_cache.bin", "junk")
        self.write("asr_cache.bin", "junk")
        findings = audit.audit_tree(self.tmp)
        cats = _categories(findings)
        self.assertIn("ocr_cache_artifact", cats)
        self.assertIn("asr_cache_artifact", cats)


# ---------------------------------------------------------------------------
# Python distribution/build metadata. Proven against the exact artifact
# `pip install .` was empirically shown to leave behind
# (src/<pkg>.egg-info/) plus the sibling categories the same install can
# produce (dist-info, build/, dist/).
# ---------------------------------------------------------------------------


class DistributionMetadataTests(TempTreeTestCase):
    def test_nested_egg_info_directory_fails(self):
        self.write("src/example.egg-info/PKG-INFO", "junk")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("egg_info_metadata", _categories(findings))
        self.assertIn("src/example.egg-info", _paths(findings))

    def test_nested_dist_info_directory_fails(self):
        self.write("src/example.dist-info/METADATA", "junk")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("dist_info_metadata", _categories(findings))
        self.assertIn("src/example.dist-info", _paths(findings))

    def test_build_directory_fails(self):
        self.write("build/lib/module.py", "junk")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("prohibited_directory", _categories(findings))
        self.assertIn("build", _paths(findings))

    def test_dist_directory_fails(self):
        self.write("dist/example-0.0.0-py3-none-any.whl", "junk")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("prohibited_directory", _categories(findings))
        self.assertIn("dist", _paths(findings))

    def test_gitignored_status_does_not_hide_the_finding(self):
        """The audit walks the raw filesystem, not git's view of it -- a
        directory being listed in .gitignore (as build/dist/*.egg-info/
        all are) must not suppress detection."""
        self.write(".gitignore", "build/\ndist/\n*.egg-info/\n")
        self.write("build/lib/module.py", "junk")
        self.write("dist/example.whl", "junk")
        self.write("src/example.egg-info/PKG-INFO", "junk")
        findings = audit.audit_tree(self.tmp)
        cats = _categories(findings)
        self.assertIn("prohibited_directory", cats)
        self.assertIn("egg_info_metadata", cats)

    def test_no_content_or_machine_path_is_printed(self):
        self.write("src/secret_example.egg-info/PKG-INFO", "sensitive metadata body\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            audit.main(["--root", str(self.tmp)])
        out = buf.getvalue()
        self.assertNotIn("sensitive metadata body", out)
        self.assertNotIn(str(self.tmp), out)


# ---------------------------------------------------------------------------
# Content patterns.
# ---------------------------------------------------------------------------


class ContentPatternTests(TempTreeTestCase):
    def test_windows_absolute_path_fails(self):
        self.write("notes.md", "See C:\\SomeFolder\\file.txt for details.\n")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("absolute_windows_drive_path", _categories(findings))

    def test_posix_home_path_fails(self):
        self.write("notes.md", "Located at /home/exampleuser/project\n")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("posix_home_path", _categories(findings))

    def test_transcript_reference_fails(self):
        token = "".join(("hacker", "rank", "_orchestrate", "_august", "26"))
        self.write("notes.md", f"path contains {token} somewhere\n")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("transcript_path_reference", _categories(findings))

    def test_secret_api_key_fixture_fails_without_printing_value(self):
        secret_value = "sk_live_" + "A" * 24
        self.write("config.py", f'API_KEY = "{secret_value}"\n')
        findings = audit.audit_tree(self.tmp)
        self.assertIn("api_key_or_token_assignment", _categories(findings))
        rendered = "\n".join(f"{f.category} {f.relative_path}" for f in findings)
        self.assertNotIn(secret_value, rendered)

    def test_organizer_style_id_fixture_fails(self):
        self.write("notes.md", "reference id msg_042 appears here\n")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("organizer_style_entity_id", _categories(findings))

    def test_internal_strategy_builder_language_fails(self):
        term = "".join(("Strat", "egy")) + " " + "Agent"
        self.write("notes.md", f"handed off to the {term} for review\n")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("internal_strategy_builder_terminology", _categories(findings))

    def test_prohibited_filename_mention_in_content_fails(self):
        self.write("notes.md", "see AGENTS.md for the ground rules\n")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("prohibited_challenge_filename_mention", _categories(findings))


# ---------------------------------------------------------------------------
# Local username: personal accounts versus CI runner accounts.
# ---------------------------------------------------------------------------

#: Fictional personal account name; never a real user.
PERSONAL_USER = "qzpersona7"


class UsernameScopeTests(TempTreeTestCase):
    """A personal account name is flagged anywhere it appears. A GitHub-hosted
    runner account ("runner" on Linux and macOS, "runneradmin" on Windows) is
    flagged only where an account name identifies an account. The current
    user is simulated, so none of this depends on who runs the suite."""

    def scan_as(self, user):
        with mock.patch.object(getpass, "getuser", return_value=user):
            return audit.audit_tree(self.tmp)

    def flagged(self, user):
        return {f.relative_path for f in self.scan_as(user) if f.category == "local_username"}

    def test_runner_accounts_do_not_flag_ordinary_runner_wording(self):
        """The hosted CI failure: ordinary prose was reported as a username."""
        self.write("docs/demo.md",
                   "The synthetic media runner supplies fixed text.\n"
                   "Runner output is fictional; two runners, the runner's log,\n"
                   "a test-runner and RUNNER in capitals are ordinary words.\n"
                   "Windows jobs run as the runneradmin account.\n")
        self.write("tests/test_example.py", '"""Exercises the example runner."""\n')
        for user in ("runner", "runneradmin", "Runner", "RUNNERADMIN"):
            with self.subTest(user=user):
                self.assertEqual(self.flagged(user), set())

    def test_personal_account_is_still_flagged_anywhere(self):
        self.write("notes.md", "Ask qzpersona7 for the shared folder.\n")
        self.write("deploy.md", "Logged in as QZPERSONA7 on the laptop.\n")
        self.write("clean.md", "Nothing personal here.\n")
        self.assertEqual(self.flagged(PERSONAL_USER), {"notes.md", "deploy.md"})

    def test_account_name_as_an_address_is_flagged(self):
        """This audit has no general email-address rule. What it checks is the
        current account name, including as the local part of an address."""
        self.write("ci.md", "Reach the build box as runner@build-host.example.\n")
        self.write("personal.md", "Write to qzpersona7@example.org for access.\n")
        self.assertIn("ci.md", self.flagged("runner"))
        self.assertIn("personal.md", self.flagged(PERSONAL_USER))

    def test_runner_account_in_sensitive_content_keeps_every_finding(self):
        self.write("linux.md", "Key copied to /home/runner/.ssh/id_rsa\n")
        self.write("tilde.md", "Config lives in ~runner/.config/tool\n")
        self.write("windows.md", "Token kept in C:\\Users\\runneradmin\\AppData\\token.txt\n")
        self.write("config.py", 'API_KEY = "runner' + "Q" * 20 + '"\n')
        self.write("key.md", "-----BEGIN RSA PRIVATE KEY-----\nrunner\n")
        linux, windows = self.scan_as("runner"), self.scan_as("runneradmin")

        def categories(findings, path):
            return {f.category for f in findings if f.relative_path == path}

        self.assertEqual(categories(linux, "linux.md"), {"posix_home_path", "local_username"})
        self.assertEqual(categories(linux, "tilde.md"), {"local_username"})
        self.assertIn("api_key_or_token_assignment", categories(linux, "config.py"))
        self.assertIn("private_key_block", categories(linux, "key.md"))
        self.assertEqual(categories(windows, "windows.md"),
                         {"absolute_windows_drive_path", "windows_userprofile_path", "local_username"})

    def test_runner_account_matching_boundaries(self):
        cases = {
            "home.md": ("/home/runner/work", True),
            "mac.md": ("/Users/Runner/work", True),
            "profile.md": ("C:\\Users\\RUNNER\\Desktop", True),
            "tilde.md": ("~runner/.cache", True),
            "address.md": ("runner@ci-host.example", True),
            "other_home.md": ("/home/runner2/work", False),
            "other_tilde.md": ("~runners/.cache", False),
            "dotted_local_part.md": ("ci.runner@ci-host.example", False),
            "longer_local_part.md": ("myrunner@ci-host.example", False),
            "dangling_at.md": ("ask the runner@ the end", False),
            "prose.md": ("the runner, a runner's job (runner).", False),
        }
        for name, (text, _) in cases.items():
            self.write(name, text + "\n")
        flagged = self.flagged("runner")
        for name, (_, expected) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(name in flagged, expected)

    def test_only_the_exact_runner_account_names_are_scoped(self):
        self.assertEqual(audit._GENERIC_CI_ACCOUNTS, frozenset({"runner", "runneradmin"}))
        self.write("notes.md", "runner2 deployed it; the runner-ci job; myrunner said so.\n")
        for user in ("runner2", "runner-ci", "myrunner"):
            with self.subTest(user=user):
                self.assertEqual(self.flagged(user), {"notes.md"})

    def test_personal_account_matching_boundaries_are_unchanged(self):
        cases = {
            "exact.md": ("qzpersona7", True),
            "upper.md": ("QZPERSONA7", True),
            "punctuated.md": ("(qzpersona7).", True),
            "underscore.md": ("qzpersona7_backup", True),
            "linux_home.md": ("/home/qzpersona7/x", True),
            "windows_profile.md": ("C:\\Users\\QzPersona7\\Documents", True),
            "address.md": ("qzpersona7@example.org", True),
            "longer.md": ("qzpersona77", False),
            "prefixed.md": ("xqzpersona7", False),
        }
        for name, (text, _) in cases.items():
            self.write(name, text + "\n")
        flagged = self.flagged(PERSONAL_USER)
        for name, (_, expected) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(name in flagged, expected)

    def test_ci_environment_variables_do_not_change_detection(self):
        """The scope follows the account name itself, never a CI flag."""
        self.write("notes.md", "Ask qzpersona7 for access; the media runner is fine.\n")
        ci_env = {"CI": "true", "GITHUB_ACTIONS": "true", "RUNNER_OS": "Linux"}
        with mock.patch.dict(os.environ, ci_env):
            self.assertEqual(self.flagged(PERSONAL_USER), {"notes.md"})
            self.assertEqual(self.flagged("runner"), set())


class RepositoryUnderRunnerAccountTests(unittest.TestCase):
    def test_repository_content_findings_are_empty_under_runner_accounts(self):
        """The exact hosted CI condition, on Linux and on Windows runners."""
        for user in ("runner", "runneradmin"):
            with self.subTest(user=user), mock.patch.object(getpass, "getuser", return_value=user):
                content_categories = {category for category, _ in audit._content_checks()}
                self.assertIn("local_username", content_categories)
                findings = audit.audit_tree(REPO_ROOT)
                content_findings = [f for f in findings if f.category in content_categories]
                self.assertEqual(content_findings, [], content_findings)


class AuditIndependenceTests(unittest.TestCase):
    def test_audit_uses_only_the_standard_library_and_no_router_code(self):
        imported = set()
        for node in ast.walk(ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("message_router", imported)
        self.assertLessEqual(imported, set(sys.stdlib_module_names) | {"__future__"})


# ---------------------------------------------------------------------------
# Symlinks.
# ---------------------------------------------------------------------------


class SymlinkTests(TempTreeTestCase):
    def test_external_symlink_fails_where_supported(self):
        outside = Path(tempfile.mkdtemp(prefix="audit_outside_")).resolve()
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        target = outside / "secret.txt"
        target.write_text("irrelevant", encoding="utf-8")
        link = self.tmp / "linked_out.txt"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("symlink privilege not available on this host")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("external_symlink", _categories(findings))

    def test_internal_symlink_is_not_flagged_as_external(self):
        real = self.write("real_dir/inner.txt", "hello")
        link = self.tmp / "linked_in.txt"
        try:
            link.symlink_to(real)
        except (OSError, NotImplementedError):
            self.skipTest("symlink privilege not available on this host")
        findings = audit.audit_tree(self.tmp)
        self.assertNotIn("external_symlink", _categories(findings))

    def test_external_symlinked_directory_is_not_traversed(self):
        outside = Path(tempfile.mkdtemp(prefix="audit_outside_dir_")).resolve()
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        (outside / "log.txt").write_text("irrelevant", encoding="utf-8")
        link = self.tmp / "linked_dir"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlink privilege not available on this host")
        findings = audit.audit_tree(self.tmp)
        self.assertIn("external_symlink", _categories(findings))
        # The prohibited file inside the external target must never be
        # reached/reported by relative path -- it was never traversed into.
        self.assertNotIn("log.txt", _paths(findings))


# ---------------------------------------------------------------------------
# Self-exemption.
# ---------------------------------------------------------------------------


class SelfExemptionTests(unittest.TestCase):
    def test_exact_two_file_exemption(self):
        self.assertEqual(
            audit.CONTENT_SCAN_EXEMPT,
            frozenset({"scripts/audit_public_tree.py", "tests/test_public_tree_audit.py"}),
        )

    def test_exemption_does_not_cover_filename_or_path_violations(self):
        tmp = Path(tempfile.mkdtemp(prefix="audit_exempt_boundary_")).resolve()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        # Even a file at the exact exempt relative path is still subject to
        # filename-pattern rules -- the exemption is content-only.
        exempt_dir = tmp / "scripts"
        exempt_dir.mkdir()
        # Simulate: same relative path, but named like a prohibited file.
        (tmp / "AGENTS.md").write_text("irrelevant", encoding="utf-8")
        findings = audit.audit_tree(tmp)
        self.assertIn("prohibited_filename", _categories(findings))

    def test_repository_content_findings_are_empty(self):
        """The two exempt files legitimately define/exercise prohibited
        content patterns; the repository's own audit must report zero
        content-pattern findings (filename/directory findings, e.g. from
        local __pycache__ build artifacts, are out of scope here)."""
        content_categories = {
            "absolute_windows_drive_path", "windows_userprofile_path", "posix_home_path",
            "transcript_path_reference", "organizer_style_entity_id",
            "internal_strategy_builder_terminology", "prohibited_challenge_filename_mention",
            "private_key_block", "aws_access_key", "api_key_or_token_assignment", "local_username",
        }
        findings = audit.audit_tree(REPO_ROOT)
        content_findings = [f for f in findings if f.category in content_categories]
        self.assertEqual(content_findings, [], content_findings)

    def test_third_file_with_identical_prohibited_content_is_detected(self):
        tmp = Path(tempfile.mkdtemp(prefix="audit_third_file_")).resolve()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "scripts").mkdir(parents=True)
        (tmp / "tests").mkdir(parents=True)
        (tmp / "other_module.py").write_text('ORGANIZER_ID_EXAMPLE = "msg_042"\n', encoding="utf-8")
        findings = audit.audit_tree(tmp)
        self.assertTrue(
            any(f.relative_path == "other_module.py" and f.category == "organizer_style_entity_id" for f in findings)
        )


# ---------------------------------------------------------------------------
# Output safety.
# ---------------------------------------------------------------------------


class OutputSafetyTests(TempTreeTestCase):
    def test_findings_report_relative_paths_only(self):
        self.write("sub/dir/log.txt", "x")
        findings = audit.audit_tree(self.tmp)
        self.assertTrue(findings)
        for f in findings:
            self.assertFalse(os.path.isabs(f.relative_path))
            self.assertNotIn(str(self.tmp), f.relative_path)

    def test_cli_output_never_prints_username_or_secret_value(self):
        secret_value = "sk_live_" + "B" * 24
        self.write("config.py", f'API_KEY = "{secret_value}"\n')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            audit.main(["--root", str(self.tmp)])
        out = buf.getvalue()
        self.assertNotIn(secret_value, out)
        current_user = getpass.getuser()
        if current_user:
            self.assertNotIn(current_user, out)

    def test_clean_tree_exits_zero(self):
        self.write("README.md", "hello\n")
        with contextlib.redirect_stdout(io.StringIO()):
            code = audit.main(["--root", str(self.tmp)])
        self.assertEqual(code, 0)

    def test_dirty_tree_exits_nonzero(self):
        self.write("log.txt", "x")
        with contextlib.redirect_stdout(io.StringIO()):
            code = audit.main(["--root", str(self.tmp)])
        self.assertNotEqual(code, 0)


# ---------------------------------------------------------------------------
# CI workflow hygiene: `pip install` and test collection both import the
# package, and each writes __pycache__/ into the source tree unless
# bytecode generation is disabled for the whole job -- which this audit
# would then correctly reject. Proven empirically: reproducing the
# committed CI sequence in a clean external clone without this setting
# failed the audit with exactly the 3 __pycache__ findings this guards.
# ---------------------------------------------------------------------------


class CIWorkflowHygieneTests(unittest.TestCase):
    def test_ci_workflow_disables_bytecode_generation_for_the_whole_job(self):
        ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        text = ci_path.read_text(encoding="utf-8")

        job_start = text.index("jobs:")
        steps_start = text.index("steps:", job_start)
        job_header = text[job_start:steps_start]

        self.assertIn(
            "PYTHONDONTWRITEBYTECODE",
            job_header,
            "PYTHONDONTWRITEBYTECODE must be set at job level (before steps:), "
            "not only inside an individual step, so it covers both the install "
            "step and test collection.",
        )

    def test_ci_installs_from_an_external_temporary_copy_of_only_pyproject_and_src(self):
        """`pip install .` (editable or not) against the checked-out repo was
        empirically shown to leave src/<pkg>.egg-info/ (and sometimes
        build/) behind. CI must instead build the install from a temp copy
        containing only pyproject.toml and src/, so nothing lands in the
        checked-out tree the audit walks."""
        ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        text = ci_path.read_text(encoding="utf-8")

        self.assertIn("tempfile.TemporaryDirectory", text)
        self.assertIn('shutil.copy2(repo_root / "pyproject.toml"', text)
        self.assertIn('shutil.copytree(repo_root / "src"', text)
        # Nothing else is copied into the temporary install source.
        self.assertNotIn("copy2(repo_root / \"README.md\"", text)
        self.assertNotIn("copytree(repo_root / \"tests\"", text)

    def test_ci_never_installs_the_repository_root_directly(self):
        """Checks only executable content, not explanatory comments (which
        legitimately mention the old `pip install .` pattern in prose)."""
        ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        executable_lines = [
            line for line in ci_path.read_text(encoding="utf-8").splitlines() if not line.strip().startswith("#")
        ]
        executable_text = "\n".join(executable_lines)
        forbidden = ("pip install .", "pip install -e .")
        for phrase in forbidden:
            self.assertNotIn(
                phrase, executable_text, f"CI must not install the checked-out repo directly: {phrase!r}"
            )

    def test_ci_contains_no_repository_cleanup_step(self):
        """The temp install copy's own cleanup (inside the `with` block, on
        a path outside the checkout) is fine; nothing may delete or reset
        anything inside the checked-out repository itself."""
        ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        text = ci_path.read_text(encoding="utf-8")
        forbidden = ("rm -rf", "git clean", "git reset --hard", "Remove-Item")
        for phrase in forbidden:
            self.assertNotIn(phrase, text, f"CI must not clean repository artifacts: {phrase!r}")

    def test_ci_runs_tests_before_the_public_tree_audit(self):
        ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        text = ci_path.read_text(encoding="utf-8")
        tests_index = text.index("unittest discover")
        audit_index = text.index("audit_public_tree.py")
        self.assertLess(tests_index, audit_index, "tests must run before the public-tree audit")


if __name__ == "__main__":
    unittest.main()
