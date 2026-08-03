"""Preparing untrusted text for pattern matching, without trusting it.

Two normalized views are produced for every piece of content:

* **folded** -- Unicode-normalized, homoglyph-mapped, case-folded, with
  whitespace collapsed. Word boundaries survive, so patterns can require
  them.
* **squeezed** -- the folded view with every non-alphanumeric character
  removed and a narrow set of digit/symbol substitutions applied. This
  catches spacing and character-substitution evasion
  (``p a s s w o r d``, ``passw0rd``) that the folded view would miss.

The squeezed view has no word boundaries at all, which makes short targets
dangerous there: a three-letter target will match inside unrelated words and
produce confident nonsense. Squeezed matching is therefore gated on a
minimum target length (:data:`SQUEEZE_MIN_LENGTH`) and is opt-in per rule.

Normalizing is not trusting. A normalized string is still untrusted
content: nothing here interprets it, and the original provenance travels
with it unchanged.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = [
    "MAX_ANALYZED_CHARS",
    "SQUEEZE_MIN_LENGTH",
    "LINE_BOUNDARY",
    "NormalizedText",
    "normalize",
    "near",
    "matches_any",
]

#: Processing bound. Content longer than this is analysed up to the limit
#: and flagged as truncated, so a very large attachment transcript cannot
#: turn matching into an unbounded amount of work.
MAX_ANALYZED_CHARS = 20_000

#: Minimum target length for squeezed matching. Below this, a target
#: matches inside ordinary words often enough to be worse than useless.
SQUEEZE_MIN_LENGTH = 5

# Characters that render like Latin letters but are not. This is the
# deception this normalizer exists to defeat; the map is deliberately
# narrow, covering the Cyrillic and Greek letters that actually appear in
# homoglyph attacks rather than every visually similar codepoint.
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p",
        "с": "c", "у": "y", "х": "x", "і": "i",
        "ѕ": "s", "һ": "h", "ԁ": "d", "к": "k",
        "ο": "o", "α": "a", "ε": "e", "ρ": "p",
        "ν": "v", "τ": "t", "κ": "k", "ι": "i",
    }
)

# Applied only to the squeezed view. Kept small on purpose: aggressive
# digit folding would rewrite ordinary numbers into letters and invent
# matches that were never in the text.
_SUBSTITUTIONS = str.maketrans({"0": "o", "1": "i", "3": "e", "5": "s", "@": "a", "$": "s"})

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

#: Internal, content-free stand-in for a line break.
#:
#: A line break is a clause boundary as real as a full stop -- people put an
#: instruction on its own line constantly -- but collapsing it into a space
#: destroys that boundary before any clause-local rule can see it, which
#: lets a framing line silently downgrade the line after it. Whitespace
#: still collapses; the boundary is kept as one deterministic marker first.
#:
#: ASCII START OF HEADING is used because it survives NFKC, case folding and
#: homoglyph mapping unchanged, is *not* matched by ``\s`` so the collapse
#: below leaves it alone, and is removed from the squeezed view by the
#: non-alphanumeric filter. The separator controls (``\x1c``-``\x1f``) look
#: like the natural choice and are not: Python classifies them as
#: whitespace, so the collapse would turn them straight back into spaces.
#: It never leaves this layer: clause segmentation splits on it, so no
#: finding, contributor, diagnostic, serialization or error can carry it.
LINE_BOUNDARY = "\x01"

#: Any run of line breaks -- LF, CRLF, CR, blank lines -- is one boundary.
#: Horizontal space is absorbed on both sides *and between* the breaks, so a
#: blank line that happens to contain spaces stays one boundary rather than
#: becoming two with an empty clause wedged between them.
_LINE_BREAK_RUN_RE = re.compile(r"[ \t\f\v]*(?:(?:\r\n|\r|\n)[ \t\f\v]*)+")

# Zero-width and bidirectional-control characters, which can hide a word
# boundary from a human reader while leaving the text machine-readable.
_INVISIBLE_RE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")


@dataclass(frozen=True)
class NormalizedText:
    """Two comparable views of one untrusted string.

    ``repr`` reports lengths only. These objects are passed around during
    analysis, and a normalized view of a message body is still the message
    body -- it must not surface in a log line or a traceback.
    """

    folded: str
    squeezed: str
    original_length: int
    truncated: bool

    @property
    def is_empty(self) -> bool:
        return not self.folded.strip()

    def __repr__(self) -> str:
        return (
            f"NormalizedText(original_length={self.original_length}, "
            f"truncated={self.truncated}, folded_length={len(self.folded)})"
        )


def normalize(text: str) -> NormalizedText:
    """Build both normalized views of ``text``."""
    original_length = len(text)
    truncated = original_length > MAX_ANALYZED_CHARS
    working = text[:MAX_ANALYZED_CHARS] if truncated else text

    # NFKC first: it resolves fullwidth and compatibility forms so the
    # homoglyph map only has to cover genuinely distinct scripts.
    working = unicodedata.normalize("NFKC", working)
    working = _INVISIBLE_RE.sub("", working)
    working = working.translate(_HOMOGLYPHS)
    # Line breaks become the boundary marker *before* whitespace collapses,
    # which is the only reason a multi-line message keeps its clauses.
    working = _LINE_BREAK_RUN_RE.sub(LINE_BOUNDARY, working)
    folded = _WHITESPACE_RE.sub(" ", working.casefold()).strip()
    folded = folded.strip(LINE_BOUNDARY + " ")

    squeezed = _NON_ALNUM_RE.sub("", folded.translate(_SUBSTITUTIONS))

    return NormalizedText(
        folded=folded,
        squeezed=squeezed,
        original_length=original_length,
        truncated=truncated,
    )


def near(left: str, right: str, gap: int = 40) -> str:
    """Build a pattern matching ``left`` and ``right`` within ``gap`` characters.

    Order-independent, because "send me your code" and "your code, send it
    over" are the same request. The bounded gap is what keeps this from
    linking two phrases that merely appear in the same long paragraph.

    The gap excludes ``.`` so a match cannot span a sentence boundary.
    Without that, the most common honest security message in existence --
    "Your code is 481920. Never share it with anyone." -- reads as a
    request, because a delivery in one sentence and a verb in the next
    look adjacent to an unbounded gap.
    """
    return f"(?:{left}[^.]{{0,{gap}}}{right}|{right}[^.]{{0,{gap}}}{left})"


def matches_any(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)
