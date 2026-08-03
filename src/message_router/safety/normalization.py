"""Preparing untrusted text for pattern matching, without trusting it.

Three normalized views are produced for every piece of content:

* **folded** -- Unicode-normalized, homoglyph-mapped, case-folded, with
  whitespace collapsed. This is the authoritative *content* view. Word
  boundaries survive, so patterns can require them, and invisible format
  characters are left in place: an emoji family sequence, Arabic joining
  behaviour and an Indic conjunct are all spelled with them.
* **security** -- the matching view. Identical to ``folded`` except that
  reviewed invisible characters are neutralised where they sit *between
  ASCII alphanumerics*, which is the only place they can split an
  ASCII-oriented safety indicator. Nothing routes, classifies, serialises
  or carries provenance from this view; it exists purely so a rule can
  match text a reader would read as one word.
* **squeezed** -- the security view with every non-alphanumeric character
  removed and a narrow set of digit/symbol substitutions applied. This
  catches spacing and character-substitution evasion
  (``p a s s w o r d``, ``passw0rd``) that the other views would miss.

The squeezed view has no word boundaries at all, which makes short targets
dangerous there: a three-letter target will match inside unrelated words and
produce confident nonsense. Squeezed matching is therefore gated on a
minimum target length (:data:`SQUEEZE_MIN_LENGTH`) and is opt-in per rule.

**Structure is typed, not spelled.** A line break is a clause boundary, and
a boundary is a fact *about* the text rather than a character *in* it. It is
therefore recorded as a :class:`StructuralBoundary` between
:class:`NormalizedSegment` records, produced only here, only from validated
newline forms. An earlier design wrote a sentinel character into the string
instead; untrusted content containing that same codepoint was then read as
trusted structure. Nothing a message can contain can add, remove, duplicate,
merge or reorder a boundary now, because content and structure no longer
share a representation.

Normalizing is not trusting. A normalized string is still untrusted
content: nothing here interprets it, and the original provenance travels
with it unchanged.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "MAX_ANALYZED_CHARS",
    "SQUEEZE_MIN_LENGTH",
    "REVIEWED_INVISIBLES",
    "StructuralBoundary",
    "NormalizedSegment",
    "NormalizedText",
    "build_security_view",
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

#: Any run of line breaks -- LF, CRLF, CR, blank lines -- is one boundary.
#: Horizontal space is absorbed on both sides *and between* the breaks, so a
#: blank line that happens to contain spaces stays one boundary rather than
#: becoming two with an empty clause wedged between them.
_LINE_BREAK_RUN_RE = re.compile(r"[ \t\f\v]*(?:(?:\r\n|\r|\n)[ \t\f\v]*)+")

#: Non-whitespace control characters, deleted from content outright.
#:
#: These carry no linguistic content in any script -- they are teleprinter
#: signalling -- so removing them is not a folding decision about language.
#: It is also what stops a control character from being mistaken for
#: structure: there is no codepoint a message can carry that survives to be
#: read as a boundary. Deliberately excludes the whitespace controls
#: (``\t \n \v \f \r`` and ``\x1c``-``\x1f``), which the collapse below
#: handles, and the two newline forms, which become typed boundaries.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0e-\x1b\x7f-\x9f]")

#: Invisible format characters reviewed for the security view.
#:
#: Every one is Unicode category ``Cf`` and renders as nothing. In content
#: they are left alone, because several are load-bearing: ``U+200D`` joins an
#: emoji family into one glyph and forms Indic conjuncts, ``U+200C`` blocks
#: Arabic joining, ``U+00AD`` is a line-break hint. In the security view they
#: are dropped only between ASCII alphanumerics -- see
#: :func:`build_security_view` -- where no script needs them and the only
#: thing they can do is split a word an ASCII-oriented rule is looking for.
REVIEWED_INVISIBLES = frozenset(
    "­"          # SOFT HYPHEN
    "​‌‍‎‏"   # ZWSP, ZWNJ, ZWJ, LRM, RLM
    "‪‫‬‭‮"   # bidi embedding / override
    "⁠⁡⁢⁣⁤"   # word joiner, invisible operators
    "⁦⁧⁨⁩"         # bidi isolates
    "﻿"          # ZERO WIDTH NO-BREAK SPACE / BOM
)


class StructuralBoundary(Enum):
    """A boundary the normalizer *decided*, not one the text spelled.

    Membership of this enum is the trust claim. A value of this type can
    only come from :func:`normalize`, so no amount of untrusted content can
    manufacture one -- which is the whole point of it being a type rather
    than a character.
    """

    LINE_BREAK = "line_break"


@dataclass(frozen=True)
class NormalizedSegment:
    """One run of content between two structural boundaries.

    ``repr`` reports offsets and lengths only: a segment of a message body
    is still the message body.
    """

    text: str
    start: int
    end: int

    def __repr__(self) -> str:
        return f"NormalizedSegment(start={self.start}, end={self.end}, length={len(self.text)})"


def build_security_view(content: str) -> str:
    """The matching view: reviewed invisibles dropped at ASCII adjacency.

    The adjacency condition is the entire safeguard. ``p<ZWSP>assword``
    hides a word from an ASCII rule and is dropped; the same character
    between two non-ASCII letters is doing typographic work in a script this
    layer has no business editing, and is kept. So an emoji ZWJ sequence, an
    Arabic ZWNJ and a Devanagari conjunct all survive unchanged, while the
    evasion does not.

    Deletion only -- never substitution, transliteration, confusable
    resolution or case change. The result is for pattern matching alone and
    never reaches a finding, a contributor, serialization or the CLI.
    """
    if not any(mark in content for mark in REVIEWED_INVISIBLES):
        return content

    kept: list[str] = []
    for index, mark in enumerate(content):
        if mark in REVIEWED_INVISIBLES:
            before = _preceding_ascii_alnum(content, index)
            after = _following_ascii_alnum(content, index)
            if before and after:
                continue
        kept.append(mark)
    return "".join(kept)


def _preceding_ascii_alnum(text: str, index: int) -> bool:
    """Is the nearest non-reviewed character before ``index`` ASCII alphanumeric?"""
    cursor = index - 1
    while cursor >= 0 and text[cursor] in REVIEWED_INVISIBLES:
        cursor -= 1
    return cursor >= 0 and _is_ascii_alnum(text[cursor])


def _following_ascii_alnum(text: str, index: int) -> bool:
    """Is the nearest non-reviewed character after ``index`` ASCII alphanumeric?"""
    cursor = index + 1
    while cursor < len(text) and text[cursor] in REVIEWED_INVISIBLES:
        cursor += 1
    return cursor < len(text) and _is_ascii_alnum(text[cursor])


def _is_ascii_alnum(mark: str) -> bool:
    return ("a" <= mark <= "z") or ("A" <= mark <= "Z") or ("0" <= mark <= "9")


@dataclass(frozen=True)
class NormalizedText:
    """The normalized views of one untrusted string, plus its structure.

    ``repr`` reports lengths only. These objects are passed around during
    analysis, and a normalized view of a message body is still the message
    body -- it must not surface in a log line or a traceback.
    """

    folded: str
    security: str
    squeezed: str
    original_length: int
    truncated: bool
    segments: tuple[NormalizedSegment, ...] = ()
    boundaries: tuple[StructuralBoundary, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.folded.strip()

    @property
    def boundary_offsets(self) -> tuple[int, ...]:
        """Security-view offsets at which a trusted boundary sits.

        One per entry in :attr:`boundaries`, pointing at the single space
        that joins two segments. Consumers split *there* rather than
        searching for a character, so there is nothing for content to spell.
        """
        return tuple(segment.end for segment in self.segments[:-1])

    def __repr__(self) -> str:
        return (
            f"NormalizedText(original_length={self.original_length}, "
            f"truncated={self.truncated}, folded_length={len(self.folded)}, "
            f"segments={len(self.segments)})"
        )


def normalize(text: str) -> NormalizedText:
    """Build the normalized views of ``text`` and its trusted structure."""
    original_length = len(text)
    truncated = original_length > MAX_ANALYZED_CHARS
    working = text[:MAX_ANALYZED_CHARS] if truncated else text

    # NFKC first: it resolves fullwidth and compatibility forms so the
    # homoglyph map only has to cover genuinely distinct scripts.
    working = unicodedata.normalize("NFKC", working)
    # Control characters go before anything else looks at the string, so no
    # later step can mistake one for structure.
    working = _CONTROL_RE.sub("", working)
    working = working.translate(_HOMOGLYPHS)

    # Split on validated newline runs *first*: this is the only place a
    # boundary is ever created, and it reads the real line breaks rather
    # than any value the message chose to contain.
    pieces = [
        _WHITESPACE_RE.sub(" ", piece.casefold()).strip()
        for piece in _LINE_BREAK_RUN_RE.split(working)
    ]
    pieces = [piece for piece in pieces if piece]

    segments: list[NormalizedSegment] = []
    cursor = 0
    for piece in pieces:
        secured = build_security_view(piece)
        segments.append(NormalizedSegment(secured, cursor, cursor + len(secured)))
        cursor += len(secured) + 1  # the joining space

    folded = " ".join(pieces)
    security = " ".join(segment.text for segment in segments)
    squeezed = _NON_ALNUM_RE.sub("", security.translate(_SUBSTITUTIONS))

    return NormalizedText(
        folded=folded,
        security=security,
        squeezed=squeezed,
        original_length=original_length,
        truncated=truncated,
        segments=tuple(segments),
        boundaries=tuple(StructuralBoundary.LINE_BREAK for _ in segments[:-1]),
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
