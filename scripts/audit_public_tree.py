"""Public-tree audit: detects content that must never leave this private
working copy before anything here is published.

Read-only. Makes no changes anywhere. Exits 0 when the scanned tree is
clean, non-zero when any prohibited filename, path component, or content
pattern is found.

Only two files are exempted from CONTENT-pattern scanning (never from
filename/path scanning): this script and its test suite, because their own
job is to define and exercise these patterns. See ``CONTENT_SCAN_EXEMPT``.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Exactly these two files. Path/filename violations still apply to them --
# only their file *content* is exempt from the content-pattern scan below.
CONTENT_SCAN_EXEMPT = frozenset(
    {
        "scripts/audit_public_tree.py",
        "tests/test_public_tree_audit.py",
    }
)

_MAX_TEXT_FILE_BYTES = 2_000_000
_TEXT_EXTENSIONS = frozenset(
    {
        ".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".yml", ".yaml",
        ".json", ".sh", ".ps1", ".cfg",
    }
)
_TEXT_BASENAMES = frozenset({".gitignore", ".gitattributes"})

# Git internals are expected structure, not a finding -- skip silently.
# (Every other cache/prohibited directory below is already caught by
# _PROHIBITED_DIR_NAMES, checked first, so this set stays minimal.)
_SILENTLY_SKIPPED_DIRS = frozenset({".git"})


@dataclass(frozen=True)
class Finding:
    relative_path: str
    category: str


# ---------------------------------------------------------------------------
# Prohibited filenames / path components (structural, not content).
# ---------------------------------------------------------------------------

_PROHIBITED_EXACT_FILENAMES = frozenset(
    {
        "log.txt",
        "AGENTS.md",
        "problem_statement.md",
        "output.csv",
        "sample_messages.csv",
        ".env",
    }
)

_PROHIBITED_FILENAME_PATTERNS = (
    ("env_variant_file", re.compile(r"^\.env\..+$")),
    ("transcript_incident_report", re.compile(r"(?i)transcript.*incident.*\.json$")),
    ("forensic_report", re.compile(r"(?i)forensic")),
    ("internal_audit_report", re.compile(r"(?i)internal.?audit")),
    ("organizer_manifest", re.compile(r"(?i)organizer.*manifest")),
    ("ocr_cache_artifact", re.compile(r"(?i)ocr.?cache")),
    ("asr_cache_artifact", re.compile(r"(?i)asr.?cache")),
    ("os_temp_artifact", re.compile(r"^(?:\.DS_Store|Thumbs\.db|desktop\.ini)$")),
    ("editor_swap_backup_file", re.compile(r".*\.(?:swp|swo|bak|orig)$")),
    ("editor_swap_backup_file", re.compile(r".*~$")),
)

_PROHIBITED_DIR_NAMES = frozenset(
    {
        "submission",
        "integrity",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "htmlcov",
        "build",
        "dist",
    }
)

# Python distribution/build metadata directories: the package-name prefix
# varies (e.g. "multimodal_message_router.egg-info"), so these are matched
# by exact suffix, never by substring, to stay precise.
_PROHIBITED_DIR_SUFFIXES = (
    ("egg_info_metadata", ".egg-info"),
    ("dist_info_metadata", ".dist-info"),
)


def _dir_suffix_category(dirname: str) -> str | None:
    for category, suffix in _PROHIBITED_DIR_SUFFIXES:
        if dirname.endswith(suffix):
            return category
    return None


_COVERAGE_FILE_RE = re.compile(r"^\.coverage(?:\..+)?$")

# ---------------------------------------------------------------------------
# Content-pattern findings. Denylist literals for organizer-specific
# proper-noun strings are built from fragments, never written whole.
# ---------------------------------------------------------------------------

_WIN_DRIVE_PATH_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
_WIN_USERPROFILE_RE = re.compile(r"%USERPROFILE%|\\Users\\[^\\\s\"'<>]+", re.IGNORECASE)
_POSIX_HOME_RE = re.compile(r"/(?:home|Users)/[^/\s\"'<>]+")

_SECRET_PATTERNS = (
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    (
        "api_key_or_token_assignment",
        re.compile(
            r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|bearer|client[_-]?secret)"
            r"\s*[:=]\s*['\"][A-Za-z0-9/+_-]{12,}"
        ),
    ),
)

_ORGANIZER_STYLE_ID_RE = re.compile(
    r"\b(?:msg|hist|sample_msg|u|img|vn|group|business)_[0-9]{2,}\b"
)

# "hackerrank" + "_orchestrate" + "_august26" assembled from fragments so the
# literal transcript-directory name never appears whole in this source file.
_TRANSCRIPT_DIR_TOKEN = "".join(("hacker", "rank", "_orchestrate", "_august", "26"))
_TRANSCRIPT_REFERENCE_RE = re.compile(re.escape(_TRANSCRIPT_DIR_TOKEN), re.IGNORECASE)

# "Strategy" + " Agent" / "Builder" + " Agent" assembled from fragments.
_STRATEGY_TERM = "".join(("Strat", "egy")) + " " + "Agent"
_BUILDER_TERM = "".join(("Build", "er")) + " " + "Agent"
_STRATEGY_BUILDER_RE = re.compile(
    "|".join(re.escape(t) for t in (_STRATEGY_TERM, _BUILDER_TERM)), re.IGNORECASE
)

_PUBLICATION_FILENAME_MENTION_RE = re.compile(r"\b(?:AGENTS\.md|problem_statement\.md)\b")


def _username_pattern() -> re.Pattern[str] | None:
    """Built at call time from the *current* OS user -- never a literal."""
    user = getpass.getuser()
    if not user or len(user) < 2:
        return None
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(user) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _content_checks() -> tuple[tuple[str, re.Pattern[str]], ...]:
    checks = [
        ("absolute_windows_drive_path", _WIN_DRIVE_PATH_RE),
        ("windows_userprofile_path", _WIN_USERPROFILE_RE),
        ("posix_home_path", _POSIX_HOME_RE),
        ("transcript_path_reference", _TRANSCRIPT_REFERENCE_RE),
        ("organizer_style_entity_id", _ORGANIZER_STYLE_ID_RE),
        ("internal_strategy_builder_terminology", _STRATEGY_BUILDER_RE),
        ("prohibited_challenge_filename_mention", _PUBLICATION_FILENAME_MENTION_RE),
    ]
    checks.extend(_SECRET_PATTERNS)
    user_pat = _username_pattern()
    if user_pat is not None:
        checks.append(("local_username", user_pat))
    return tuple(checks)


# ---------------------------------------------------------------------------
# Walking the tree.
# ---------------------------------------------------------------------------


def _is_text_eligible(path: Path) -> bool:
    if path.name in _TEXT_BASENAMES:
        return True
    return path.suffix.lower() in _TEXT_EXTENSIONS


def _relpath(path: Path, root: Path) -> str:
    return PurePosixPath(path.relative_to(root)).as_posix()


def audit_tree(root: Path) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    content_checks = _content_checks()

    for dirpath_str, dirnames, filenames in os.walk(root, followlinks=False):
        dirpath = Path(dirpath_str)

        # Symlinked subdirectories: never descended into (followlinks=False
        # already guarantees this); report only if they resolve outside the
        # repository, then always prune them from further traversal.
        kept_dirnames = []
        for d in list(dirnames):
            entry = dirpath / d
            if entry.is_symlink():
                try:
                    target = entry.resolve(strict=False)
                    target.relative_to(root)
                    inside = True
                except (ValueError, OSError):
                    inside = False
                if not inside:
                    findings.append(Finding(_relpath(entry, root), "external_symlink"))
                continue  # never traverse into a symlinked directory
            if d in _PROHIBITED_DIR_NAMES:
                findings.append(Finding(_relpath(entry, root), "prohibited_directory"))
                continue  # flagged; do not descend further
            suffix_category = _dir_suffix_category(d)
            if suffix_category is not None:
                findings.append(Finding(_relpath(entry, root), suffix_category))
                continue  # flagged; do not descend further
            if d in _SILENTLY_SKIPPED_DIRS:
                continue  # .git: expected structure, not a finding
            kept_dirnames.append(d)
        dirnames[:] = kept_dirnames

        for fname in filenames:
            fpath = dirpath / fname
            rel = _relpath(fpath, root)

            if fpath.is_symlink():
                try:
                    target = fpath.resolve(strict=False)
                    target.relative_to(root)
                    inside = True
                except (ValueError, OSError):
                    inside = False
                if not inside:
                    findings.append(Finding(rel, "external_symlink"))
                    continue

            if fname in _PROHIBITED_EXACT_FILENAMES:
                findings.append(Finding(rel, "prohibited_filename"))
                continue
            if _COVERAGE_FILE_RE.match(fname):
                findings.append(Finding(rel, "coverage_cache_artifact"))
                continue
            matched_pattern = False
            for category, pattern in _PROHIBITED_FILENAME_PATTERNS:
                if pattern.search(fname):
                    findings.append(Finding(rel, category))
                    matched_pattern = True
                    break
            if matched_pattern:
                continue

            if rel in CONTENT_SCAN_EXEMPT:
                continue
            if not _is_text_eligible(fpath):
                continue
            try:
                if fpath.stat().st_size > _MAX_TEXT_FILE_BYTES:
                    continue
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            for category, pattern in content_checks:
                if pattern.search(text):
                    findings.append(Finding(rel, category))

    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit this repository tree for content that must not be published."
    )
    parser.add_argument(
        "--root",
        default=None,
        help="directory to audit (default: the repository containing this script)",
    )
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else Path(__file__).resolve().parents[1]
    findings = audit_tree(root)

    if not findings:
        print("result=PASS")
        print("findings=0")
        return 0

    print("result=FAIL")
    print(f"findings={len(findings)}")
    for finding in sorted(findings, key=lambda f: (f.relative_path, f.category)):
        print(f"  {finding.category}: {finding.relative_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
