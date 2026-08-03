"""Resolving a media source string to a file inside an explicit root.

A document supplies media as a *relative* source string. That string is
untrusted input like any other, so it is checked before it is ever joined
to a real directory, and the joined result is then proved to still be
inside the root rather than assumed to be.

Errors name the rule that was broken. They never contain the source
string, the root, or the resolved location -- a rejected path is exactly
the kind of value that should not end up in a log line.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

__all__ = ["PathRejection", "MediaPathError", "MediaRootResolver"]


class PathRejection(Enum):
    """Why a media source string was refused."""

    EMPTY = "empty_source"
    NUL_BYTE = "nul_byte"
    ABSOLUTE = "absolute_path"
    DRIVE_QUALIFIED = "drive_qualified_path"
    BACKSLASH_SEPARATOR = "backslash_separator"
    TRAVERSAL = "traversal_segment"
    ESCAPES_ROOT = "escapes_media_root"
    SYMLINK_ESCAPES_ROOT = "symlink_escapes_media_root"
    NOT_FOUND = "not_found"
    NOT_A_REGULAR_FILE = "not_a_regular_file"


class MediaPathError(Exception):
    """A media source string could not be safely resolved.

    Carries a :class:`PathRejection` only. Deterministic and value-free.
    """

    def __init__(self, rejection: PathRejection) -> None:
        self.rejection = rejection
        super().__init__(rejection.value)

    def __str__(self) -> str:
        return self.rejection.value

    def __repr__(self) -> str:
        return f"MediaPathError(rejection={self.rejection.value!r})"


@dataclass(frozen=True)
class MediaRootResolver:
    """Resolves relative source strings beneath one explicit directory.

    The root is supplied by the caller rather than discovered, so nothing
    here can wander into a directory the caller did not nominate.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve(strict=False))

    def resolve(self, source: str) -> Path:
        """Return the file ``source`` names inside :attr:`root`.

        Raises :class:`MediaPathError` if the string is not a plain
        relative path, if it would leave the root, if any component is a
        symlink pointing outside, or if the target is missing or is not a
        regular file.
        """
        relative = self._validate_source(source)
        candidate = self.root
        for part in relative.parts:
            candidate = candidate / part
            # Checked per component: a symlinked *directory* partway along
            # the path would otherwise redirect everything after it before
            # the final containment test ever runs.
            if candidate.is_symlink() and not self._contains(candidate.resolve(strict=False)):
                raise MediaPathError(PathRejection.SYMLINK_ESCAPES_ROOT)

        resolved = candidate.resolve(strict=False)
        if not self._contains(resolved):
            raise MediaPathError(PathRejection.ESCAPES_ROOT)
        if not resolved.exists():
            raise MediaPathError(PathRejection.NOT_FOUND)
        if not resolved.is_file():
            raise MediaPathError(PathRejection.NOT_A_REGULAR_FILE)
        return resolved

    # -- internals --------------------------------------------------------

    def _validate_source(self, source: str) -> PurePosixPath:
        if not isinstance(source, str) or not source.strip():
            raise MediaPathError(PathRejection.EMPTY)
        if "\x00" in source:
            raise MediaPathError(PathRejection.NUL_BYTE)
        if "\\" in source:
            # Sources are POSIX-style so one string means one thing on every
            # platform; a backslash would be a separator on one OS and a
            # legal filename character on another.
            raise MediaPathError(PathRejection.BACKSLASH_SEPARATOR)
        if len(source) >= 2 and source[1] == ":" and source[0].isalpha():
            raise MediaPathError(PathRejection.DRIVE_QUALIFIED)
        if source.startswith("/"):
            raise MediaPathError(PathRejection.ABSOLUTE)

        relative = PurePosixPath(source)
        if relative.is_absolute():
            raise MediaPathError(PathRejection.ABSOLUTE)
        if any(part == ".." for part in relative.parts):
            raise MediaPathError(PathRejection.TRAVERSAL)
        if not relative.parts:
            raise MediaPathError(PathRejection.EMPTY)
        return relative

    def _contains(self, resolved: Path) -> bool:
        try:
            return os.path.commonpath([str(resolved), str(self.root)]) == str(self.root)
        except ValueError:
            # Raised when the two paths sit on different drives, which is
            # itself proof the candidate is outside the root.
            return False
