"""Deterministic indicator rules over normalized untrusted content.

Every rule is a regular expression or a small composition of them. There is
no model, no learned threshold, and no network call anywhere in this file:
given the same content, the same rules fire, every time.

**Composition over keywords.** A bare keyword is almost never enough. The
word "password" appears in a phishing attempt, in a bank's own warning not
to share one, in a receipt, and in a security-training slide. Rules
therefore combine a *target* with a *request directed at the recipient*,
within a bounded distance, and then consult the surrounding context before
deciding how strongly to report.

**Context classes** (:class:`ContextFlags`) are what keep the honest cases
quiet:

* *warning* -- "we will never ask for your code" is advice, not a request;
* *notification* -- "your code is ..." delivers something, it does not ask;
* *educational* -- security training discusses attacks without performing one;
* *quoted* -- a user forwarding something suspicious to ask about it.

**Known limits.** Coverage is English-shaped and pattern-based. Paraphrase,
other languages, and novel phrasings will be missed; these rules raise
signals, they do not certify anything. A message with no findings has not
been proved safe -- it has only failed to match a finite list.
"""

from __future__ import annotations

import ipaddress
import re
import urllib.parse
from dataclasses import dataclass, field

from ..domain.enums import Reliability
from .models import DiagnosticCode, FindingCategory, Severity
from .normalization import NormalizedText, SQUEEZE_MIN_LENGTH, near

__all__ = ["RuleHit", "ContextFlags", "classify_context", "evaluate_content", "evaluate_propagation"]


@dataclass(frozen=True)
class RuleHit:
    """A rule fired. Provenance and source are attached later by the analyzer."""

    category: FindingCategory
    severity: Severity
    rule_id: str
    diagnostic: DiagnosticCode
    ambiguous: bool = False
    indicators: tuple[str, ...] = ()
    reliability: Reliability = Reliability.MEDIUM


def _c(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def _compile_all(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(_c(p) for p in patterns)


def _any(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(p.search(text) for p in patterns)


# ---------------------------------------------------------------------------
# Phrase-head grammar, shared by the rules that must know what a verb governs.
# ---------------------------------------------------------------------------
#
# "stop taking your medication" and "stop the prescription delivery" both put
# a treatment word within a few tokens of a cessation verb, but only the
# first is *about* the treatment: in the second, "prescription" modifies
# "delivery", and the head of the phrase is an administrative noun. The same
# shape decides "send me your PIN" against "send me the connector pins".
#
# Heads are identified positively. A token is the head of its phrase when
# what follows is a boundary or a **closed-class** word -- determiner,
# preposition, conjunction, adverb. The alternative would be enumerating the
# nouns that disqualify it, and those are an open class: delivery, shipment,
# dispatch, auto-refill; connector, dowel, cotter, retaining. Whatever a
# pharmacy or a hardware catalogue names next is already covered, because the
# rule never asks "is this an administrative noun?" or "is this a component?"
# -- it asks whether the sensitive word is the head, and any unlisted
# open-class follower answers no.
_FUNCTION_WORD = (
    r"(?:without|unless|before|after|until|while|when|since|because|so|and|or|but|"
    r"then|now|today|tonight|tomorrow|immediately|straight|right|at|from|to|into|on|"
    r"in|with|for|of|by|as|if|even|though|entirely|completely|altogether|permanently|"
    r"temporarily|forever|again|anyway|soon|please|thanks|thank|ok|okay|here|back|"
    r"over|via|through|that|this|it|i|we|you|they|he|she|my|your|our|their|"
    # A numeral does not change which word is the head -- "your PIN 4417" and
    # "your medication #2" are still about the credential and the medication --
    # and adverbs are closed enough to admit by shape.
    r"#?\d[\d,.-]*|\w+ly)"
)
#: The governed token is the head of its phrase: end of input, punctuation,
#: or a closed-class word follows it. The punctuation class is closed and
#: deliberately admits no letters or digits, so it can never carry the match
#: through into an unrelated noun phrase.
_NP_HEAD_END = (
    r"(?=$|\s*[.,;:!?)\]}\(\[{\"'–—/-]|\s+" + _FUNCTION_WORD + r"\b)"
)

#: A bounded run of determiners, possessives, quantities and treatment
#: qualifiers that may sit between a verb and the noun it governs. Every
#: member is closed-class, so this run can never swallow an open-class noun
#: and shift which word is the head.
_QUANTITY = r"(?:\d+|#\d+|one|two|three|four|five|six|seven|eight|nine|ten|both|several)"
_NP_PREFIX = (
    r"(?:\s+(?:the|a|an|your|my|our|their|his|her|its|this|that|these|those|any|all|"
    r"of|more|further|remaining|prescribed|prescription|repeat|regular|daily|"
    r"current|new|old|usual|" + _QUANTITY + r"))*"
)
#: The same run without first-person possessives.
#:
#: "Stop my prescription" is a request a person makes to a pharmacy about
#: their own treatment; "stop your medication" is advice aimed at the reader.
#: The approved severe policy requires direction *toward the recipient*, so
#: the direct-object path -- where the cessation verb governs the treatment
#: with no consumption action in between -- does not accept "my" or "our".
#: The consumption path keeps them: "stop taking my prescribed medication"
#: is still an instruction about taking it.
_NP_PREFIX_DIRECTED = (
    r"(?:\s+(?:the|a|an|your|their|his|her|its|this|that|these|those|any|all|"
    r"of|more|further|remaining|prescribed|prescription|repeat|regular|daily|"
    r"current|new|old|usual|" + _QUANTITY + r"))*"
)


# ---------------------------------------------------------------------------
# Context classification.
# ---------------------------------------------------------------------------

#: Every bounded action the credential layer treats as asking for a secret.
#:
#: One list, used to build both the request grammar and the warning grammar
#: below, because two hand-maintained lists drift and the drift is silent in
#: the worst possible direction: a verb present in the request list and
#: missing from the warning list turns "never confirm your PIN over the
#: phone" -- an anti-fraud warning -- into the demand it warns against.
#: A test asserts the containment rather than trusting it.
_DISCLOSURE_ACTIONS = (
    "reply with", "read out", "hand over", "send", "share", "give", "tell",
    "forward", "provide", "submit", "enter", "type", "confirm", "verify",
    "text", "paste", "reveal", "disclose", "transmit",
)

#: Transmission actions. A request only inside the bounded
#: message-to-recipient grammar, but a *warning* in any form: "never message
#: support your PIN" prohibits exactly what that grammar detects.
_TRANSMISSION_ACTIONS = ("message", "dm", "whatsapp")


def _alternation(actions: tuple[str, ...]) -> str:
    """An alternation with the longest phrases first, so they win the match."""
    return "(?:" + "|".join(sorted(actions, key=len, reverse=True)) + ")"


#: The protective form must cover every action the request form accepts.
_WARNING_ACTION = _alternation(_DISCLOSURE_ACTIONS + _TRANSMISSION_ACTIONS)

_WARNING_PATTERNS = _compile_all(
    # The object matters. "Never share your code" is a security warning;
    # "don't tell your doctor" and "don't tell anyone" are not, and treating
    # them as advisories would silence the unsafe-advice and secrecy rules
    # that exist precisely to catch them.
    r"\b(?:never|do ?n[o']?t|dont|don t)\b[^.]{0,40}"
    r"\b" + _WARNING_ACTION + r"\b[^.]{0,30}"
    r"\b(?:otp|pins?|personal identification numbers?|password|passcode|code|codes|"
    r"card|cvv|credentials|login|details|account details|security answer)\b",
    r"\bwe (?:will )?never (?:ask|request|call|email|text)\b",
    # Protective framing without a negator. "Report anyone asking you to
    # send your PIN" prohibits the request by telling the reader what to do
    # about it, so requiring "never" would miss a standard bank warning.
    r"\breport (?:anyone|anybody|any (?:caller|message|person|number)|it|them|this)\b"
    r"[^.]{0,40}\b(?:asking|asks|who asks|that asks|requesting|demanding)\b",
    r"\bno ?one (?:from|at)\b[^.]{0,40}\bwill (?:ever )?ask\b",
    r"\b(?:scam|phishing|fraud) (?:alert|warning|attempt)\b",
    r"\bbeware of\b",
    r"\bstay (?:safe|alert)\b[^.]{0,30}\b(?:scam|fraud|phishing)\b",
    r"\bhow to (?:spot|avoid|recognise|recognize|report)\b[^.]{0,20}\b(?:scam|phishing|fraud)\b",
)

_NOTIFICATION_PATTERNS = _compile_all(
    r"\byour (?:otp|pin|code|password|verification code|login code|security code) is\b",
    r"\b(?:here is|this is) your (?:otp|pin|code|verification code|login code)\b",
    r"\buse (?:this|the following) code\b",
    r"\b(?:otp|verification code|login code)\s*[:=]\s*",
    r"\bwe (?:have )?(?:generated|sent|issued) (?:a|your) (?:code|otp|password)\b",
    r"\b(?:receipt|invoice) (?:no|number|#)\b",
    r"\bpayment (?:received|confirmed|successful)\b",
    r"\bthank you for your (?:payment|order|purchase)\b",
    r"\byour (?:order|booking|appointment|statement)\b",
)

_EDUCATIONAL_PATTERNS = _compile_all(
    r"\battackers?\b[^.]{0,40}\b(?:often|typically|usually|may|might|will)\b",
    r"\b(?:for example|e\.?g\.?)\b[^.]{0,40}\b(?:scam|phish|attack|fraud)\b",
    r"\b(?:security )?(?:awareness|training|course|module|lesson|workshop)\b",
    r"\bthis (?:article|guide|post|training|module|course)\b",
    r"\b(?:explains?|teaches?|demonstrates?) how\b",
    r"\bin (?:this|the) (?:example|demo|simulation|exercise)\b",
)

_QUOTED_PATTERNS = _compile_all(
    r"\bi (?:just )?(?:got|received)\b[^.]{0,40}\b(?:message|text|sms|email|call)\b",
    r"\bsomeone (?:sent|texted|messaged|emailed)\b",
    r"\b(?:this|the) message (?:said|says|claims|reads)\b",
    r"\bis this (?:a )?(?:scam|real|legit|genuine|phishing)\b",
    r"\bshould i (?:trust|believe|reply to)\b",
    r"\bforwarded from\b",
)


# A warning whose object is a pronoun: "Your code is 1234. Never share it
# with anyone." The noun sits in an earlier sentence, so the patterns above
# -- which need the noun after the verb -- miss it. Requiring a secret noun
# somewhere in the content keeps this from matching "don't send it to him".
_PRONOUN_WARNING_RE = _c(
    r"\b(?:never|do ?n[o']?t|dont)\b[^.]{0,20}"
    r"\b(?:share|give|send|disclose|reveal|forward|tell)\b[^.]{0,15}"
    r"\b(?:it|them|this|that|these|those)\b"
)
_SECRET_NOUN_RE = _c(
    r"(?:otp|one[\s-]?time[\s-]?(?:password|code|pin)|\bpins?\b|"
    r"personal identification numbers?|passcode|password|"
    r"verification code|login code|security code|access code|recovery code|"
    r"authentication code|auth code|2fa code|\bcode\b|cvv|card number)"
)


def _is_pronoun_warning(text: str) -> bool:
    return bool(_SECRET_NOUN_RE.search(text)) and bool(_PRONOUN_WARNING_RE.search(text))


@dataclass(frozen=True)
class ContextFlags:
    """Surrounding-context classes for one piece of content."""

    warning: bool = False
    notification: bool = False
    educational: bool = False
    quoted: bool = False

    @property
    def is_advisory(self) -> bool:
        """True when the content is telling someone *not* to be caught out.

        A warning is the opposite of a request, so a request-shaped rule is
        silenced entirely rather than merely downgraded.
        """
        return self.warning

    @property
    def is_discussion(self) -> bool:
        """True when the content is talking about attacks rather than making one."""
        return self.educational or self.quoted


def classify_context(norm: NormalizedText) -> ContextFlags:
    """Document-level framing. Retained for auditability only.

    Every rule gates on :class:`_LocalContext` instead. A sentence about
    what attackers do says nothing about the next sentence, and treating it
    as though it did turns "this article explains phishing" into a prefix
    that downgrades whatever follows it.
    """
    text = norm.security
    return ContextFlags(
        warning=_any(_WARNING_PATTERNS, text) or _is_pronoun_warning(text),
        notification=_any(_NOTIFICATION_PATTERNS, text),
        educational=_any(_EDUCATIONAL_PATTERNS, text),
        quoted=_any(_QUOTED_PATTERNS, text),
    )


#: Strong clause boundaries spelled *in* the text. Deliberately not a
#: parser: sentence terminators, semicolons and colons only. Line breaks are
#: absent on purpose -- they arrive as typed boundaries from normalization,
#: which is what stops untrusted content from spelling one.
_CLAUSE_BOUNDARY_RE = re.compile(r"[.!?;:]+")

#: Weak separators: a comma followed by a transition word, or one of a few
#: conjunctions that routinely join two independent clauses.
#:
#: These are *candidates*, never boundaries on their own. "Doctors and
#: nurses", "the PIN and account number", and an ordinary list comma all
#: contain these tokens and must not split. A candidate becomes a boundary
#: only when both sides independently carry structure worth evaluating --
#: see :func:`_side_is_independent` -- which is a structural test over the
#: existing framing and safety patterns, not an inference about meaning.
_WEAK_SEPARATOR_RE = _c(
    r",\s+(?:but|however|yet|nevertheless|instead|then|therefore|still)\b"
    r"|\s+(?:but|yet|and then|while|although)\b"
)


def _side_is_independent(side: str) -> bool:
    """True when a side carries framing or an independently matchable rule.

    Structural only: it asks whether an existing pattern fires, never what
    the words mean. Requiring this on *both* sides is what keeps a noun
    phrase like "doctors and nurses" or a coordinated object like "the PIN
    and account number" from being torn in half.
    """
    stripped = side.strip()
    if len(stripped) < 3:
        return False
    flags = _flags_for(stripped)
    if flags.warning or flags.notification or flags.educational or flags.quoted:
        return True
    # A completed quotation is a bounded span with two ends, so it is
    # independently evaluable whatever it contains. Without this a
    # quotation-only side fails the gate, the boundary is refused, and the
    # quote's span swallows the unquoted clause on the other side of it.
    if _quoted_spans(stripped):
        return True
    if _any(_ADVICE_CRITICAL_CONTEXT, stripped):
        return True
    for _rid, pattern, _diag, _neg in _UNSAFE_COMPILED:
        if pattern.search(stripped):
            return True
    for _rid, pattern, _diag in _INJECTION_COMPILED:
        if pattern.search(stripped):
            return True
    for _rid, request_pattern, _target, _diag in _CREDENTIAL_COMPILED:
        if request_pattern.search(stripped):
            return True
    if _MESSAGE_TRANSMISSION_RE.search(stripped):
        return True
    # The financial, identity and pressure families count too. Without them
    # a side carrying nothing but a coercive demand was not "independently
    # evaluable", the boundary in front of it was refused, and a warning or
    # a receipt on the other side of a comma absorbed it.
    return any(
        _any(patterns, stripped)
        for patterns in (
            _PAYMENT_DEMAND, _THREAT, _OFF_CHANNEL, _SECRECY,
            _ANTI_VERIFICATION, _ADVANCE_FEE, _IDENTITY_CLAIM, _TIME_PRESSURE,
            # Routine billing is its own construction too: an invoice clause
            # must be separable from whatever follows the comma, or it
            # supplies "ordinary" cover for a demand that is not.
            _ORDINARY_BILLING, _ORDINARY_DEADLINE,
        )
    )


def _segment(text: str, boundaries: tuple[int, ...] = ()) -> list[tuple[int, int]]:
    """Clause spans as ``(start, end)``, strong boundaries then weak ones.

    ``boundaries`` are the *typed* offsets normalization produced from real
    line breaks. They are handed in rather than searched for, because a
    boundary the caller can find in the text is a boundary the message can
    write. Punctuation cuts are still read from the text -- a full stop is a
    full stop whoever typed it -- but a line break is a claim about
    structure, and only the normalizer is allowed to make one.
    """
    cuts = [(match.start(), match.end()) for match in _CLAUSE_BOUNDARY_RE.finditer(text)]
    # A typed boundary is the single joining space between two segments.
    cuts.extend((offset, offset + 1) for offset in boundaries)
    cuts.sort()

    spans: list[tuple[int, int]] = []
    start = 0
    for cut_start, cut_end in cuts:
        if cut_start < start:          # already consumed by an adjacent cut
            start = max(start, cut_end)
            continue
        spans.append((start, cut_start))
        start = cut_end
    spans.append((start, len(text)))

    refined: list[tuple[int, int]] = []
    for span_start, span_end in spans:
        cursor = span_start
        for match in _WEAK_SEPARATOR_RE.finditer(text, span_start, span_end):
            left, right = text[cursor:match.start()], text[match.end():span_end]
            if _side_is_independent(left) and _side_is_independent(right):
                refined.append((cursor, match.start()))
                cursor = match.end()
        refined.append((cursor, span_end))
    return refined
#: A clause that is itself a quotation.
_QUOTE_ENCLOSED_RE = _c(r"^[\"'“‘(\[]|[\"'”’)\]]$")

# ---------------------------------------------------------------------------
# Quotation attribution.
#
# Quotation marks are punctuation, not provenance. Anyone can type them, so
# treating a balanced pair as protective framing hands out an exemption for
# the price of two characters -- a demand wrapped in quotes is still a
# demand. A quoted span is protected only when something structurally
# attached to it says what the quote is *for*: a warning, a criticism, a
# lesson, a document, a report of what someone said, or a hypothetical.
#
# Each family below is a bounded, enumerated construction. None of them
# infers anything from the sender, their display name, or the relationship.
# ---------------------------------------------------------------------------

#: Warning or criticism about the quoted content.
_ATTR_WARNING = (
    r"do not (?:follow|obey|comply|act on|respond to|reply to)|never (?:follow|obey|comply)"
    r"|this is (?:unsafe|dangerous|fraudulent|a scam|a phishing|phishing|fake|not genuine)"
    r"|beware of|report (?:messages|anything|any message|it|this|them)"
    r"|(?:warned|warns|warning) (?:that|about)|(?:a|the|this|common) (?:scam|phishing|fraud)"
    r"|do not (?:fall|be fooled)"
)
#: Educational or documentary framing.
_ATTR_EDUCATIONAL = (
    r"(?:an|one|another|this) example of|(?:for|as an) example"
    r"|documentation (?:states|says|shows)|training (?:explains|says|shows|quotes)"
    r"|(?:the )?lesson (?:discusses|explains)|(?:the )?article (?:describes|explains|quotes)"
    r"|quoted for awareness|the following is an example|awareness (?:example|training)"
    r"|(?:module|course|guide|bulletin|leaflet) (?:covers|explains|quotes|says)"
)
#: Historical reporting or reported speech.
_ATTR_REPORTED = (
    r"(?:previously|earlier|once) (?:said|wrote|claimed)"
    r"|(?:the |a )?(?:report|incident|bulletin|log|record) (?:recorded|noted|described|involved)"
    r"|(?:said|says|saying|reads|read|wrote|writes|asked|asks|claims|claimed|replied|told)"
    r"|(?:message|text|sms|email|note|post|caller|attacker|scammer|sender) "
)
#: Hypothetical framing.
_ATTR_HYPOTHETICAL = (
    r"suppose (?:someone|somebody|a message)|imagine (?:receiving|getting|a message)"
    r"|a hypothetical|(?:if|when) (?:someone|somebody) (?:says|writes|asks|sends)"
    r"|what if (?:someone|somebody)"
)
_ATTRIBUTION = (
    r"(?:" + _ATTR_WARNING + r"|" + _ATTR_EDUCATIONAL
    + r"|" + _ATTR_REPORTED + r"|" + _ATTR_HYPOTHETICAL + r")"
)

#: Attribution introducing the quote, within a bounded window immediately
#: before the opening mark. "Greetings, then a demand in quotes" is not
#: attribution, which is why the window is short and stops at a full stop.
_ATTRIBUTION_BEFORE_RE = _c(_ATTRIBUTION + r"[^.!?]{0,40}$")

#: Attribution following the quote -- "<demand>", do not follow it.
_ATTRIBUTION_AFTER_RE = _c(r"^[^.!?]{0,40}" + _ATTRIBUTION)


def _attributed_quoted_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Quoted spans that something structurally attached explains.

    A completed quotation with no attribution on either side is left
    unprotected: it is reported speech only if somebody reports it.
    """
    attributed: list[tuple[int, int]] = []
    for start, end in _quoted_spans(text):
        before = text[:start]
        after = text[end + 1:]
        if _ATTRIBUTION_BEFORE_RE.search(before) or _ATTRIBUTION_AFTER_RE.search(after):
            attributed.append((start, end))
    return tuple(attributed)


def _has_attribution(clause: str) -> bool:
    """True when this clause frames a quotation it contains or introduces."""
    return bool(_c(_ATTRIBUTION).search(clause))


class _LocalContext:
    """Framing resolved per clause, with one narrow inheritance rule.

    A clause carries its own warning, notification, educational and quoted
    flags. The single exception is a clause that *is* a quotation — enclosed
    in quotes, or introduced by a preceding clause that announces one — which
    inherits the quoted and educational framing of the clause that framed it.
    That is the "narrowly associated clause" case, and it is the only way
    framing crosses a boundary.
    """

    __slots__ = ("_spans", "_flags", "_fallback", "_quoted", "_line_starts")

    def __init__(self, text: str, boundaries: tuple[int, ...] = ()) -> None:
        # Where a real line begins, taken from the typed boundaries rather
        # than searched for. A list marker is only a marker at the start of a
        # line, and "the start of a line" is a fact the message cannot spell.
        self._line_starts = frozenset({0} | {offset + 1 for offset in boundaries})
        self._spans = [
            (start, end, text[start:end]) for start, end in _segment(text, boundaries)
        ]
        self._fallback = _flags_for(text)
        # Attributed spans only: quote marks on their own protect nothing.
        self._quoted = _attributed_quoted_spans(text)

        raw = [_flags_for(clause) for _, _, clause in self._spans]
        enclosed = _enclosing_quotation(tuple(clause for _, _, clause in self._spans))
        resolved: list[ContextFlags] = []
        for index, flags in enumerate(raw):
            clause = self._spans[index][2].strip()
            # A quotation that runs past a line break is still one quotation.
            # While it stays open, the clauses inside it keep the framing of
            # the clause that opened it -- but only if that clause actually
            # framed it. An unattributed quotation inherits nothing.
            introduced = index and _has_attribution(self._spans[index - 1][2])
            if index and introduced and (enclosed[index] or _QUOTE_ENCLOSED_RE.search(clause)):
                previous = resolved[index - 1]
                flags = ContextFlags(
                    warning=flags.warning or previous.warning,
                    notification=flags.notification,
                    educational=flags.educational or previous.educational,
                    quoted=flags.quoted or previous.quoted or enclosed[index],
                )
            resolved.append(flags)
        self._flags = resolved

    @property
    def bounds(self) -> tuple[tuple[int, int], ...]:
        """The resolved clause spans, for scanning one clause at a time."""
        return tuple((start, end) for start, end, _clause in self._spans)

    def index_at(self, position: int) -> int:
        """Which clause a match sits in. ``-1`` when it sits outside them all."""
        for index, (start, end, _clause) in enumerate(self._spans):
            if start <= position <= end:
                return index
        return -1

    def clause(self, index: int) -> str:
        return self._spans[index][2] if 0 <= index < len(self._spans) else ""

    def introduces_list(self, index: int, text: str) -> bool:
        """True when this clause ended on a colon, i.e. it announces a list."""
        if not 0 <= index < len(self._spans):
            return False
        end = self._spans[index][1]
        return end < len(text) and text[end] == ":"

    def is_numbered_marker(self, index: int, text: str) -> bool:
        """True when this clause is only the ``1.`` marker of a list item.

        Deliberately narrow, because a full stop after a digit is far more
        often a decimal, a version, a date, a money amount or a house number
        than a list. All of these must hold: the clause is nothing but one or
        two digits, it begins a real line -- one the *normalizer* recorded,
        never one the content spelled -- it is terminated by the full stop
        that split it off, and what follows does not continue a number.
        ``1)`` never reaches here: no boundary splits it, so it stays a
        bullet the ordinary item pattern already matches.
        """
        if not 0 <= index < len(self._spans):
            return False
        start, end, clause = self._spans[index]
        stripped = clause.strip()
        if not (1 <= len(stripped) <= 2 and stripped.isdigit()):
            return False
        if start not in self._line_starts:
            return False
        if not (end < len(text) and text[end] == "."):
            return False
        body = self.clause(index + 1).strip()
        return bool(body) and not body[0].isdigit()

    def clause_at(self, position: int) -> str:
        """The clause a match sits in, for suppressors that must be local."""
        for start, end, clause in self._spans:
            if start <= position <= end:
                return clause
        return ""

    def at(self, position: int) -> ContextFlags:
        """Framing in force where a rule actually matched."""
        flags = self._fallback
        for index, (start, end, _clause) in enumerate(self._spans):
            if start <= position <= end:
                flags = self._flags[index]
                break
        # A quotation inside a clause frames what is inside it and nothing
        # else: "a scam message reads '<demand>' -- report it" is one clause,
        # so clause-level framing alone cannot tell the report from the demand.
        for start, end in self._quoted:
            if start <= position <= end:
                return ContextFlags(
                    warning=flags.warning,
                    notification=flags.notification,
                    educational=flags.educational,
                    quoted=True,
                )
        return flags


#: Marks that open and close a quotation. Apostrophes are deliberately
#: excluded from both: ``'`` and ``’`` are contraction and possessive marks
#: far more often than quotes, and counting them would let "we don't share
#: codes" open a quotation that protects everything after it.
_OPEN_QUOTES = "\"“"
_CLOSE_QUOTES = "\"”"


def _quote_open_after(clause: str, open_now: bool) -> bool:
    """Whether a quotation is still open once this clause has been read."""
    for mark in clause:
        if open_now:
            if mark in _CLOSE_QUOTES:
                open_now = False
        elif mark in _OPEN_QUOTES:
            open_now = True
    return open_now


def _enclosing_quotation(clauses: tuple[str, ...]) -> tuple[bool, ...]:
    """Which clauses sit inside a quotation that actually closes.

    The closure requirement is the safeguard. A quotation is a span with two
    ends; a lone stray mark is punctuation. Without this, one unmatched
    quote character would silently protect every clause after it, which is a
    bypass an attacker can type. So an unterminated opening protects nothing.
    """
    open_after: list[bool] = []
    state = False
    for clause in clauses:
        state = _quote_open_after(clause, state)
        open_after.append(state)

    terminates = [False] * len(clauses)
    closed_later = False
    for index in range(len(clauses) - 1, -1, -1):
        if not open_after[index]:
            closed_later = True
        terminates[index] = closed_later

    return tuple(
        index > 0 and open_after[index - 1] and terminates[index]
        for index in range(len(clauses))
    )


def _clause_scan(
    pattern: re.Pattern[str], text: str, local: _LocalContext
) -> list[re.Match[str]]:
    """Every match of ``pattern``, each confined to a single clause.

    Scanning the whole string looks equivalent and is not, for two reasons.
    A match may *start* in a framed clause and run into an actionable one,
    and it is then judged by the framing where it started. And because
    ``finditer`` yields non-overlapping matches, that first match consumes
    the real request, so no second match is ever offered for the clause that
    mattered. Bounding each scan to one clause makes both impossible, and
    makes ``local.at(match.start())`` exact by construction.
    """
    found: list[re.Match[str]] = []
    for start, end in local.bounds:
        found.extend(pattern.finditer(text, start, end))
    return found


def _quoted_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Character ranges enclosed by a quotation that actually closes.

    An opening mark with no partner yields nothing, for the reason given in
    :func:`_enclosing_quotation`: otherwise a single typed quote character
    would frame the whole rest of the message.
    """
    spans: list[tuple[int, int]] = []
    start_mark: int | None = None
    for index, mark in enumerate(text):
        if start_mark is None:
            if mark in _OPEN_QUOTES:
                start_mark = index
        elif mark in _CLOSE_QUOTES:
            spans.append((start_mark, index))
            start_mark = None
    return tuple(spans)


def _flags_for(clause: str) -> ContextFlags:
    return ContextFlags(
        warning=_any(_WARNING_PATTERNS, clause) or _is_pronoun_warning(clause),
        notification=_any(_NOTIFICATION_PATTERNS, clause),
        educational=_any(_EDUCATIONAL_PATTERNS, clause),
        quoted=_any(_QUOTED_PATTERNS, clause),
    )


# ---------------------------------------------------------------------------
# Prompt / instruction injection.
# ---------------------------------------------------------------------------

_INJECTION_RULES: tuple[tuple[str, str, DiagnosticCode], ...] = (
    (
        # The object list deliberately excludes "message". "Ignore my
        # previous message, wrong chat" is a retraction -- one of the most
        # common things anyone sends -- and treating it as an override
        # attempt would veto ordinary conversation. An override attempt has
        # to name instructions, rules, or a prompt.
        "INJ_OVERRIDE_PRIOR",
        r"\b(?:ignore|disregard|forget|override|bypass)\b[^.]{0,30}"
        r"\b(?:previous|prior|earlier|above|all|any|system|these|your)\b[^.]{0,20}"
        r"\b(?:instruction|instructions|rule|rules|prompt|prompts|direction|directions|guideline|guidelines)\b",
        DiagnosticCode.OVERRIDE_INSTRUCTION,
    ),
    (
        "INJ_FORCE_ACTION",
        r"\b(?:mark|set|treat|classify|route|flag|label)\b[^.]{0,25}"
        r"\b(?:as\s+)?(?:interrupt|defer|suppress)\b"
        r"|\baction\s*[:=]\s*(?:interrupt|defer|suppress)\b",
        DiagnosticCode.FORCED_OUTCOME_INSTRUCTION,
    ),
    (
        "INJ_FORCE_CONFIDENCE",
        r"\bconfidence\s*[:=]\s*[0-9]",
        DiagnosticCode.FORCED_OUTCOME_INSTRUCTION,
    ),
    (
        "INJ_ASSERT_TRUST",
        r"\b(?:verified_sender|is_verified|trusted_sender|sender_verified)\s*[:=]\s*(?:true|1|yes)\b"
        r"|\bthis (?:message|sender) is (?:verified|trusted|whitelisted)\b",
        DiagnosticCode.ASSERTED_TRUST_IN_CONTENT,
    ),
    (
        "INJ_DISABLE_SAFETY",
        r"\b(?:disable|turn off|skip|bypass|suspend|ignore)\b[^.]{0,25}"
        r"\b(?:safety|security|filter|filters|moderation|check|checks|screening|guardrail|guardrails)\b",
        DiagnosticCode.SAFETY_SUPPRESSION_INSTRUCTION,
    ),
    (
        "INJ_SYSTEM_DISCLOSURE",
        r"\b(?:reveal|show|print|repeat|output|dump)\b[^.]{0,25}"
        r"\b(?:system prompt|your prompt|your instructions|initial instructions|configuration|api key|secret key)\b",
        DiagnosticCode.SYSTEM_DISCLOSURE_INSTRUCTION,
    ),
    (
        "INJ_ROLE_OVERRIDE",
        r"\byou are (?:now|no longer)\b|\b(?:act|behave|respond) as (?:if|though|a|an)\b"
        r"|\bpretend (?:to be|you are)\b|\bnew (?:system )?instructions?\s*[:=]",
        DiagnosticCode.OVERRIDE_INSTRUCTION,
    ),
)

_INJECTION_COMPILED = tuple((rid, _c(pat), diag) for rid, pat, diag in _INJECTION_RULES)


def _evaluate_injection(norm: NormalizedText, local: _LocalContext) -> list[RuleHit]:
    hits: list[RuleHit] = []
    for rule_id, pattern, diagnostic in _INJECTION_COMPILED:
        contexts = [local.at(m.start()) for m in _clause_scan(pattern, norm.security, local)]
        if not contexts:
            continue
        # An instruction that stands unframed somewhere is an instruction,
        # however it was framed elsewhere.
        if all(c.is_discussion for c in contexts):
            # Describing an attack is not performing one. Still reported --
            # a quoted injection is worth seeing -- but never severe.
            hits.append(
                RuleHit(
                    FindingCategory.PROMPT_INJECTION,
                    Severity.SUSPICIOUS,
                    rule_id,
                    DiagnosticCode.QUOTED_OR_EDUCATIONAL_CONTEXT,
                    ambiguous=True,
                )
            )
            continue
        hits.append(
            RuleHit(FindingCategory.PROMPT_INJECTION, Severity.SEVERE, rule_id, diagnostic)
        )
    return hits


# ---------------------------------------------------------------------------
# Credential solicitation.
# ---------------------------------------------------------------------------

#: Built from the shared action list, so the request grammar and the
#: protective warning grammar cannot drift apart. Adding an action here adds
#: its warning form in the same edit.
_REQUEST_VERB = _alternation(_DISCLOSURE_ACTIONS)
_RECIPIENT_POSSESSIVE = r"(?:your|the|me|us|it)"

# A bare "pin" is a credential in one sense and a sewing supply, a map
# marker, a connector, a firing pin, a pinned message, or a variable name in
# many others. Those senses are an **open class** -- connector, dowel, cotter,
# retaining, tuning, whatever component someone names next -- so listing the
# ones that are not credentials can never be finished.
#
# The token is therefore admitted by what *is* there rather than by what is
# not: either the unambiguous full form, or a "pin" whose phrase head it
# actually is -- a determiner or possessive immediately before it, optionally
# with one approved credential modifier in between. An unapproved modifier
# breaks that adjacency and the phrase is simply not a credential target, no
# matter which modifier it is. "Pinned" and "pinning" never match at all,
# because the token boundary rejects them.
_PIN_CRED_MODIFIER = (
    r"(?:account|security|banking|bank|card|debit[ -]card|credit[ -]card|atm|"
    r"authentication|auth|access|verification|login|sign[ -]?in)"
)
_PIN_DETERMINER = (
    r"(?:the|a|an|this|that|your|my|our|their|his|her|its|"
    r"recipient'?s?|customer'?s?)"
)
_PIN_TARGET = (
    r"(?:personal identification numbers?"
    r"|\b" + _PIN_DETERMINER + r"\s+(?:" + _PIN_CRED_MODIFIER + r"\s+)?"
    r"pin(?:'s|s)?\b" + _NP_HEAD_END + r")"
)

_SECRET_TARGET = (
    r"(?:otp|one[\s-]?time[\s-]?(?:password|code|pin)|" + _PIN_TARGET + r"|passcode|password|"
    r"verification code|login code|security code|access code|recovery code|"
    r"authentication code|auth code|2fa code|security answer|secret question)"
)
_CARD_TARGET = r"(?:cvv|cvc|card number|card details|security code on the back|expiry date|full card)"
_BANK_TARGET = (
    r"(?:net ?banking|online banking|internet banking|account login|banking (?:login|password|credentials)|"
    r"account credentials|bank credentials|upi pin|atm pin)"
)

_CREDENTIAL_FAMILIES = (
    ("CRED_SECRET_DISCLOSURE", _SECRET_TARGET, DiagnosticCode.CREDENTIAL_DISCLOSURE_REQUEST),
    ("CRED_CARD_DISCLOSURE", _CARD_TARGET, DiagnosticCode.CARD_DETAIL_REQUEST),
    ("CRED_BANKING_DISCLOSURE", _BANK_TARGET, DiagnosticCode.BANKING_ACCESS_REQUEST),
)

_CREDENTIAL_COMPILED = tuple(
    (
        rule_id,
        _c(near(f"{_REQUEST_VERB}(?:\\s+{_RECIPIENT_POSSESSIVE})?", target, gap=30)),
        _c(target),
        diagnostic,
    )
    for rule_id, target, diagnostic in _CREDENTIAL_FAMILIES
)

# Squeezed fallback: catches spacing/substitution evasion for long targets
# only. Short targets such as a three-letter one are excluded because with
# no word boundaries they match inside ordinary words.
_SQUEEZED_TARGETS = _compile_all(
    r"onetimepassword", r"verificationcode", r"logincode", r"securitycode",
    r"passcode", r"password", r"recoverycode", r"authenticationcode",
    r"netbanking", r"onlinebanking", r"cardnumber",
)
_SQUEEZED_REQUEST = _compile_all(
    r"sendme", r"shareyour", r"giveme", r"tellme", r"replywith", r"forwardthe",
    r"provideyour", r"enteryour", r"typeyour", r"sendyour", r"sendthe",
)

# Messaging a credential to someone is a disclosure request, but "message"
# is a noun at least as often as a verb -- "check the message for your PIN"
# must stay quiet. So the action is admitted only when it governs *both* a
# bounded recipient and the credential being transmitted; a bare "message"
# never enters the general request-verb list.
_MESSAGE_ACTION = r"(?:direct[- ]?message|message|dm|text|whatsapp)"
#: A reviewed, closed set of support-desk destinations. Deliberately not
#: open: a display name, a company, or an arbitrary noun must never become a
#: recipient, or "send the connector pins to the vendor" would read as a
#: credential request. Determiners and natural plurals are allowed; nothing
#: beyond this list is inferred.
#: A reviewed qualifier and a reviewed head. Both lists are closed, so the
#: cross product is closed too -- it admits "customer support team" and
#: "account representative" without ever inferring a noun that is not in
#: one of these two lists.
_ROLE_QUALIFIER = r"(?:customer (?:support|service)|customer|support|account|service|help ?)"
_ROLE_HEAD = r"(?:representatives?|agents?|teams?|desks?|operators?|admins?)"
_TRANSMISSION_ROLE = (
    r"(?:(?:" + _ROLE_QUALIFIER + r"\s*)?" + _ROLE_HEAD
    + r"|customer (?:support|service)|support)"
)
_TRANSMISSION_DESTINATION = (
    r"(?:me|us|him|her|them|(?:the|our|your|their)\s+" + _TRANSMISSION_ROLE
    + r"|" + _TRANSMISSION_ROLE + r")"
)
#: The spelled-out credential may carry a determiner or possessive; the
#: short forms already carry their own.
_FULL_FORM_CREDENTIAL = (
    r"(?:(?:the|your|my|our|their|a)\s+)?personal identification numbers?"
)
_TRANSMITTED_CREDENTIAL = r"(?:" + _FULL_FORM_CREDENTIAL + r"|" + _SECRET_TARGET + r")"

_MESSAGE_TRANSMISSION_RE = _c(
    # message me your PIN  /  direct-message the service desk your PIN
    r"\b" + _MESSAGE_ACTION + r"\s+" + _TRANSMISSION_DESTINATION + r"\s+(?:with\s+)?"
    + _TRANSMITTED_CREDENTIAL
    # message the account PIN to the support representative
    + r"|\b" + _MESSAGE_ACTION + r"\s+" + _TRANSMITTED_CREDENTIAL
    + r"[^.]{0,20}\bto\s+" + _TRANSMISSION_DESTINATION + r"\b"
    # send support a message with your security PIN
    + r"|\bsend\s+(?:" + _TRANSMISSION_DESTINATION + r"\s+)?an?\s+"
    + _MESSAGE_ACTION + r"\s+(?:with|containing|including|holding)\s+"
    + _TRANSMITTED_CREDENTIAL
    # send a message to support with your security PIN
    + r"|\bsend\s+an?\s+" + _MESSAGE_ACTION + r"\s+to\s+" + _TRANSMISSION_DESTINATION
    + r"\s+(?:with|containing|including|holding)\s+" + _TRANSMITTED_CREDENTIAL
)


def _evaluate_credentials(norm: NormalizedText, local: _LocalContext) -> list[RuleHit]:
    text = norm.security
    hits: list[RuleHit] = []

    for rule_id, request_pattern, target_pattern, diagnostic in _CREDENTIAL_COMPILED:
        target_contexts = [local.at(m.start()) for m in _clause_scan(target_pattern, text, local)]
        request_contexts = [local.at(m.start()) for m in _clause_scan(request_pattern, text, local)]
        target_present = bool(target_contexts)

        if not target_present:
            continue

        # Framing is read where each occurrence actually sits, and the
        # strongest unframed one decides. A warning in one sentence says
        # nothing about a request in the next.
        unframed_asks = [c for c in request_contexts
                          if not c.is_advisory and not c.is_discussion]
        framed_asks = [c for c in request_contexts if c.is_discussion]
        request_present = bool(unframed_asks or framed_asks)
        if unframed_asks:
            ctx = unframed_asks[0]
        elif framed_asks:
            ctx = framed_asks[0]
        else:
            ctx = next((c for c in target_contexts
                        if not c.is_advisory and not c.is_discussion), target_contexts[0])

        # An advisory telling someone never to share a secret is the
        # opposite of asking for one. Silenced outright.
        if ctx.is_advisory:
            continue

        if request_present:
            if ctx.is_discussion:
                hits.append(
                    RuleHit(
                        FindingCategory.CREDENTIAL_SOLICITATION,
                        Severity.WEAK,
                        rule_id,
                        DiagnosticCode.QUOTED_OR_EDUCATIONAL_CONTEXT,
                        ambiguous=True,
                    )
                )
                continue
            # A notification does not suppress an actual request: "your code
            # is X, now send it to me" is still a request.
            hits.append(
                RuleHit(FindingCategory.CREDENTIAL_SOLICITATION, Severity.SEVERE, rule_id, diagnostic)
            )
            continue

        if ctx.notification or ctx.is_discussion:
            # Delivering or discussing a code, with nothing asked of anyone.
            continue

        # The target was named with no request attached -- worth noting,
        # nothing more.
        hits.append(
            RuleHit(
                FindingCategory.CREDENTIAL_SOLICITATION,
                Severity.WEAK,
                f"{rule_id}_MENTION",
                DiagnosticCode.CREDENTIAL_DISCLOSURE_REQUEST,
                ambiguous=True,
            )
        )

    # A bounded message-to-recipient transmission request. Kept separate from
    # the general request verbs on purpose: a bare "message" would flag "check
    # the message for your PIN", so the action has to govern both a recipient
    # and the credential being transmitted.
    for match in _clause_scan(_MESSAGE_TRANSMISSION_RE, text, local):
        ctx = local.at(match.start())
        if ctx.is_advisory:
            continue
        if ctx.is_discussion:
            hits.append(
                RuleHit(
                    FindingCategory.CREDENTIAL_SOLICITATION,
                    Severity.WEAK,
                    "CRED_MESSAGE_TRANSMISSION",
                    DiagnosticCode.QUOTED_OR_EDUCATIONAL_CONTEXT,
                    ambiguous=True,
                )
            )
            continue
        hits.append(
            RuleHit(
                FindingCategory.CREDENTIAL_SOLICITATION,
                Severity.SEVERE,
                "CRED_MESSAGE_TRANSMISSION",
                DiagnosticCode.CREDENTIAL_DISCLOSURE_REQUEST,
            )
        )
        break

    squeezed_ctx = local.at(0)
    if (not hits and not squeezed_ctx.is_advisory and not squeezed_ctx.is_discussion
            and not squeezed_ctx.notification):
        squeezed = norm.squeezed
        if len(squeezed) >= SQUEEZE_MIN_LENGTH and _any(_SQUEEZED_TARGETS, squeezed) and _any(
            _SQUEEZED_REQUEST, squeezed
        ):
            hits.append(
                RuleHit(
                    FindingCategory.CREDENTIAL_SOLICITATION,
                    Severity.SEVERE,
                    "CRED_OBFUSCATED_DISCLOSURE",
                    DiagnosticCode.CREDENTIAL_DISCLOSURE_REQUEST,
                )
            )
    return hits


# ---------------------------------------------------------------------------
# Payment, coercion, advance-fee.
# ---------------------------------------------------------------------------

_PAYMENT_DEMAND = _compile_all(
    r"\b(?:pay|send|transfer|deposit|remit|wire)\b[^.]{0,25}"
    r"\b(?:now|immediately|today|urgently|at once|right away|within)\b",
    r"\b(?:make|complete) (?:the |a )?payment\b[^.]{0,20}\b(?:now|immediately|today|urgent)\b",
    r"\boutstanding (?:amount|balance)\b[^.]{0,25}\b(?:immediately|now|today)\b",
)
_OFF_CHANNEL = _compile_all(
    r"\bgift ?cards?\b", r"\bitunes card\b", r"\bsteam card\b", r"\bgoogle play card\b",
    r"\b(?:bitcoin|btc|ethereum|crypto|usdt|wallet address)\b",
    r"\b(?:western union|moneygram|wire transfer)\b",
    r"\b(?:pay|send)\b[^.]{0,20}\bto this (?:wallet|account|number|upi)\b",
)
_THREAT = _compile_all(
    r"\b(?:account|service|policy|subscription)\b[^.]{0,30}"
    r"\b(?:will be |be )?(?:suspended|closed|terminated|blocked|deactivated|frozen)\b",
    r"\b(?:legal action|lawsuit|arrest|police|court|prosecution|fine|penalty)\b",
    r"\byou will (?:lose|be charged|be reported|be disconnected)\b",
    r"\bfinal (?:notice|warning|reminder before)\b",
)
_SECRECY = _compile_all(
    r"\b(?:do ?n[o']?t|dont|never)\b[^.]{0,20}\btell (?:anyone|anybody|your family|your spouse)\b",
    r"\bkeep this (?:confidential|secret|between us|to yourself)\b",
    r"\bthis is (?:strictly )?confidential\b[^.]{0,30}\b(?:do ?n[o']?t|dont) (?:share|discuss)\b",
)
_ANTI_VERIFICATION = _compile_all(
    r"\b(?:do ?n[o']?t|dont|no need to)\b[^.]{0,30}"
    r"\b(?:call|contact|verify with|check with|confirm with)\b[^.]{0,25}"
    r"\b(?:bank|support|helpline|customer (?:care|service)|official)\b",
    r"\b(?:do ?n[o']?t|dont)\b[^.]{0,20}\b(?:verify|confirm|double ?check)\b",
    r"\bthere is no need to (?:verify|confirm|check)\b",
)
_ADVANCE_FEE = _compile_all(
    r"\b(?:you have )?(?:won|win|winner of|selected for)\b[^.]{0,40}"
    r"\b(?:prize|lottery|reward|gift|cash|award)\b",
    r"\b(?:refund|rebate|compensation|claim)\b[^.]{0,40}"
    r"\b(?:pay|fee|charge|processing|handling|release|clearance)\b",
    r"\b(?:pay|send)\b[^.]{0,25}\b(?:processing|handling|clearance|release|delivery) fee\b",
)
_ORDINARY_BILLING = _compile_all(
    r"\binvoice\b", r"\breceipt\b", r"\bstatement\b", r"\bdue date\b",
    r"\byour (?:order|subscription|booking|plan)\b", r"\bpayment (?:received|confirmed)\b",
    r"\bautopay\b", r"\bdirect debit\b",
)


@dataclass(frozen=True)
class _Occurrence:
    """Where one indicator matched: the clause it sits in and that clause's
    framing. ``clause`` never leaves this layer -- it exists so the bounded
    relationship grammar below can read the words that link two clauses."""

    index: int
    clause: str
    flags: ContextFlags


@dataclass(frozen=True)
class _Signal:
    """Every occurrence of one indicator family, with where each one sits.

    Collect-then-aggregate. Reading framing from a single anchor match made
    the whole family answer to whichever occurrence happened to come first:
    one advisory clause silenced every later demand, and one educational
    clause capped the severity of the whole message. Each occurrence is now
    judged where it sits, and the family is decided from all of them.
    """

    occurrences: tuple[_Occurrence, ...]
    unframed: bool

    def __bool__(self) -> bool:
        # Truthy when at least one occurrence stands outside advisory framing.
        return bool(self.occurrences)


def _collect_signal(
    patterns: tuple[re.Pattern[str], ...], text: str, local: _LocalContext,
    confined: bool = True,
) -> _Signal:
    """Every non-advisory occurrence of ``patterns``, in deterministic order.

    Order is pattern order, then clause, then match position -- all fixed,
    so the result never depends on iteration accidents.

    ``confined`` bounds each scan to one clause, which is what makes an
    occurrence's clause index mean anything. Scanning the whole string lets a
    pattern *start* in one clause and run into the next: the match is then
    filed under the clause it began in, and because ``finditer`` yields
    non-overlapping matches it also swallows the real occurrence further on.
    "Invoice 8841 is attached; your account will be suspended" filed the
    threat under the invoice and lost it in the demand's own clause.

    It is switched off only for a rule whose approved form is already a
    cross-clause relationship -- a prize named in one clause and its release
    fee in the next.
    """
    occurrences: list[_Occurrence] = []
    unframed = False
    for pattern in patterns:
        spans = local.bounds if confined else ((0, len(text)),)
        for start, end in spans:
            for match in pattern.finditer(text, start, end):
                ctx = local.at(match.start())
                if ctx.is_advisory:
                    # An advisory occurrence says nothing about the others.
                    continue
                occurrences.append(
                    _Occurrence(local.index_at(match.start()),
                                local.clause_at(match.start()), ctx)
                )
                if not ctx.is_discussion:
                    unframed = True
    return _Signal(tuple(occurrences), unframed)


def _is_ordinary(signal: _Signal) -> bool:
    """True when *every* occurrence of a family sits in routine billing.

    Requiring all of them is the point: an invoice in one clause must not
    make a coercive demand in another read as routine, and a receipt that
    arrives after a threat must not erase it.
    """
    return bool(signal) and all(
        _any(_ORDINARY_BILLING, occurrence.clause) or occurrence.flags.notification
        for occurrence in signal.occurrences
    )


# ---------------------------------------------------------------------------
# Binding coercive signals to a payment.
#
# A threat and a gift-card mention in two unrelated clauses used to satisfy
# the coercion rules between them, because the rules asked only whether both
# families appeared *somewhere*. "The vendor threatened legal action in 2019.
# Gift cards are on the shelf." is not a scam, and nothing in the message
# said it was -- the two halves were never connected to each other.
#
# A combined finding now has to name the payment it is about. The anchor is
# an independently supported payment demand, and the coercive signal has to
# reach it through one of five bounded, enumerated relationships. Proximity
# is not one of them: adjacency alone, shared topic, shared paragraph and
# document co-presence all fail. Where the grammar cannot prove a link, both
# findings survive independently and nothing is combined -- interpreting the
# link is left to semantic analysis this layer deliberately does not attempt.
# ---------------------------------------------------------------------------

#: The financial nouns a reference can be about.
_REFERENCE_OBJECT = r"(?:payment|fee|amount|transfer|charge|invoice|balance|sum)"

#: The determiner an action's object must carry, and the bounded run of
#: qualifiers that may sit between them. The determiner is required, not
#: optional: without it "wire transfer" -- an ordinary compound noun -- reads
#: as the verb "wire" governing "transfer", and every mention of a wire
#: transfer becomes a payment instruction. Every member is closed-class, so
#: the run can never carry the match into an unrelated noun phrase.
_REFERENCE_DET = r"(?:the|this|that|a|an|your|our|their|my|its)"
_REFERENCE_QUAL = (
    r"(?:\s+(?:required|outstanding|full|remaining|same|above|attached|"
    r"following|first|final|total|agreed|monthly|annual))*"
)

#: The payment actions a reference may perform. Enumerated, and deliberately
#: not extended: this is the vocabulary that says money is being moved.
_PAYMENT_ACTION = (
    r"(?:pay|paying|send|sending|transfer|transferring|remit|remitting|settle|settling|"
    r"submit|submitting|complete|completing|make|making|wire|wiring|clear|clearing)"
)

#: A bounded predicate saying that the named payment is to be made. Reached
#: only across a closed run of auxiliaries, so it cannot be borrowed from a
#: different subject later in the clause.
_PAYMENT_PREDICATE = (
    r"(?:goes? by|goes? through|paid|payable|settled|remitted|cleared|sent|transferred|due)"
)
_REFERENCE_AUX = (
    r"(?:\s+(?:must|should|shall|will|would|can|could|may|is|are|was|were|be|being|been|"
    r"to|now|then|only|also|still|already))*"
)

#: An adjacent clause pointing back at the anchored payment.
#:
#: A definite financial noun on its own is *not* a reference. "Parse the
#: invoice", "archive the charge" and "summarize the payment notice" all name
#: a payment and do nothing to it, and accepting them let an unrelated
#: document-handling clause next to a real demand pull in a gift-card mention
#: and suppress the message. Eligibility is therefore a positive grammar --
#: an approved payment action governing the object, a bounded predicate
#: saying the object is to be paid, a pronoun standing in for one, or an
#: explicit payment purpose -- rather than a list of verbs to exclude.
_PAYMENT_REFERENCE_RE = _c(
    # An approved payment action governing a payment object.
    r"\b" + _PAYMENT_ACTION + r"\b\s+" + _REFERENCE_DET + _REFERENCE_QUAL
    + r"\s+" + _REFERENCE_OBJECT + r"\b"
    # A named payment the clause says is to be paid.
    r"|\b(?:this|that|the|your|our|my) " + _REFERENCE_OBJECT + r"\b"
    + _REFERENCE_AUX + r"\s+" + _PAYMENT_PREDICATE + r"\b"
    # A pronoun standing in for the payment.
    r"|\b(?:send|pay|transfer|complete|settle) it\b"
    r"|\bdo so\b"
    # An explicit payment purpose.
    r"|\b(?:to|for) (?:pay|settle|complete|send|transfer|remit)\b"
    r"|\b(?:to|for) (?:this|that|the|your|our|my) " + _REFERENCE_OBJECT + r"\b"
)

#: An adjacent clause continuing the anchored payment instruction.
#:
#: Bare "using", "via" and "through" are deliberately absent. They are three
#: of the commonest prepositions in English and govern whatever follows them,
#: which is usually not a payment: "the parcel went through the depot", "we
#: fixed the leak using a spanner". Accepting them on their own attached an
#: unrelated adjacent clause to a real payment demand and turned a deferral
#: into a suppression. Every continuation below either is a consequence
#: connective or names the payment action it continues.
_COERCIVE_CONTINUATION_RE = _c(
    r"\botherwise\b|\bor else\b|\bfailing that\b"
    r"|\bpay (?:with|using|via)\b|\bsend (?:through|via)\b"
    r"|\bto (?:avoid|prevent|stop)\b[^.]{0,30}\b(?:penalty|closure|suspension|charge|fee|action)\b"
    r"|\bbefore the (?:deadline|cut ?off|due date)\b"
)

#: The bounded payment objects two clauses may share. Definite on purpose:
#: two clauses both saying "payment" in the abstract share a topic, not a
#: transaction, and topic is exactly what must not be enough.
_PAYMENT_OBJECT_RE = _c(
    r"\b(?:the|this|that) (payment|fee|transfer|invoice|charge|balance)\b"
)

#: What distinguishes one transaction from another of the same kind.
#:
#: A definite head noun alone does not name a transaction. "The payment for
#: March" and "the payment for April" are two payments, "the fee" and "the
#: fee we refunded" are two fees, and a rule that compared only the head
#: joined them -- so an ordinary instalment notice read as one coercive
#: episode. Each branch below opens with a **closed** token set, and only the
#: single content token it introduces is compared. Nothing is stored: the
#: pair is built, compared, and discarded inside this module.
_OBJECT_QUALIFIER_RE = _c(
    # A restrictive attachment naming what the transaction is for.
    r"^\s+(?:for|from|of|regarding|covering|towards?)\s+"
    r"(?:the|a|an|our|your|their|this|that)?\s*([a-z0-9]+)"
    # A relative clause naming what happened to it.
    r"|^\s+(?:we|you|they|he|she|i|who|which|that)\s+([a-z0-9]+)"
)

#: A bounded past-period qualifier, which distinguishes a settled transaction
#: from the one being demanded. Enumerated, and read only within a short
#: window after the object so it cannot be borrowed from a later clause.
_PAST_PERIOD_RE = _c(
    r"^[^.]{0,40}?\b(?:last (?:year|week|month|quarter|term|season)"
    r"|previously|earlier|back in \d{4}|in (?:19|20)\d{2})\b"
)

#: A payment action governing a definite payment object. This is the bounded
#: construction that re-establishes an episode past the adjacent clause --
#: naming the object again is not enough, something has to be done to it.
_PAYMENT_DIRECTED_RE = _c(
    r"\b(?:pay|send|transfer|remit|settle|complete|make|wire)\b"
    r"(?:\s+(?:the|this|that|a|an|your|our|their|it))*\s*"
    r"\b(?:payment|fee|transfer|invoice|charge|balance)\b"
    r"|\b(?:send|pay|transfer|complete|settle) it\b"
    r"|\bto (?:pay|settle)\b"
    r"|\bdo so\b"
)

#: A bounded list item, used only to keep an anchored group together.
_LIST_ITEM_RE = _c(r"^\s*(?:[-*•–—·]+|\(?\d{1,2}[.)])\s+")

def _payment_objects(clause: str) -> frozenset[tuple[str, str]]:
    """Every definite payment object in ``clause``, as (family, qualifier).

    ``qualifier`` is ``""`` when nothing distinguishes the transaction, and
    otherwise a single bounded token drawn from the closed constructions in
    :data:`_OBJECT_QUALIFIER_RE` and :data:`_PAST_PERIOD_RE`.
    """
    objects: set[tuple[str, str]] = set()
    for match in _PAYMENT_OBJECT_RE.finditer(clause):
        tail = clause[match.end():]
        qualifier = ""
        attached = _OBJECT_QUALIFIER_RE.match(tail)
        if attached is not None:
            qualifier = next(group for group in attached.groups() if group)
        elif _PAST_PERIOD_RE.match(tail):
            qualifier = "past-period"
        objects.add((match.group(1), qualifier))
    return frozenset(objects)


def _same_payment_episode(left: str, right: str) -> bool:
    """True when both clauses name one transaction rather than two.

    Requires the same head-noun family *and* the same qualifier family. A
    qualified object and an unqualified one are not assumed to be the same
    payment: where the grammar cannot tell, it declines to join them.
    """
    left_objects, right_objects = _payment_objects(left), _payment_objects(right)
    if not (left_objects and right_objects):
        return False
    return any(
        family == other_family and qualifier == other_qualifier
        for family, qualifier in left_objects
        for other_family, other_qualifier in right_objects
    )


def _distinct_episode(left: str, right: str) -> bool:
    """True when an explicit qualifier says the two clauses are two transactions.

    A veto on the reference relationship: "pay the invoice for March" beside
    "the invoice for April" is two invoices, and adjacency must not make it
    one. Two *unqualified* objects are not distinguished -- "the balance" and
    "the payment" are ordinarily the same demand named twice -- so this asks
    only whether something explicit separates them: the same head noun
    carrying different qualifiers, or two qualified objects that disagree.
    """
    left_objects, right_objects = _payment_objects(left), _payment_objects(right)
    if not (left_objects and right_objects):
        return False
    if _same_payment_episode(left, right):
        return False
    shared = {family for family, _q in left_objects} & {family for family, _q in right_objects}
    if shared:
        return True  # one kind of object, two different transactions of it
    return all(qualifier for _f, qualifier in left_objects) and all(
        qualifier for _f, qualifier in right_objects
    )


def _in_anchored_list(anchor: int, other: int, local: _LocalContext, text: str) -> bool:
    """True when ``other`` is a list item in the group ``anchor`` introduces.

    The group is the unbroken run of list items immediately following a
    clause that ends on a colon. The first non-item clause closes it, so a
    later unrelated bullet inherits nothing.

    A numbered marker written ``1.`` is split off by the clause boundary
    before this ever sees it -- the marker becomes a clause of its own and
    the item body loses its bullet -- so a numbered instruction inside a
    real payment group was dropped while the same list written ``1)`` or
    ``-`` was kept. The marker clause is recognised here (see
    :meth:`_LocalContext.is_numbered_marker`, which reads only trusted line
    starts) and the clause it introduces counts as that item's body.
    """
    if other <= anchor or not local.introduces_list(anchor, text):
        return False
    after_marker = False
    for index in range(anchor + 1, other + 1):
        clause = local.clause(index)
        if not clause.strip():
            # The colon and the line break leave an empty clause between the
            # anchor and its first item; it is punctuation, not a member.
            continue
        if local.is_numbered_marker(index, text):
            after_marker = True
            continue
        if _LIST_ITEM_RE.match(clause) or after_marker:
            # A body only ever answers to the marker directly in front of it.
            after_marker = False
            continue
        return False
    return True


def _relationship(
    anchor: _Occurrence, other: _Occurrence, local: _LocalContext, text: str
) -> str | None:
    """The bounded relationship binding ``other`` to a payment anchor, if any.

    Every relationship past the anchor's own clause is bounded twice over:
    structurally, to the adjacent clause or the anchor's own list group, and
    referentially, to one transaction. A generic financial head shared by two
    clauses is not a relationship -- see :func:`_same_payment_episode`.
    """
    if anchor.index < 0 or other.index < 0:
        return None
    if anchor.index == other.index:
        return "REL_SAME_CLAUSE"
    if _in_anchored_list(anchor.index, other.index, local, text):
        return "REL_ANCHORED_LIST"
    if abs(anchor.index - other.index) == 1:
        # Adjacency is necessary, never sufficient: a reference that names a
        # *different* transaction is refused even next door.
        if _distinct_episode(anchor.clause, other.clause):
            return None
        if _PAYMENT_REFERENCE_RE.search(other.clause):
            return "REL_EXPLICIT_REFERENCE"
        if _COERCIVE_CONTINUATION_RE.search(other.clause):
            return "REL_CONTINUATION"
        # Naming the same payment object in the next clause is not, by
        # itself, a relationship either -- "pay the invoice now. archive the
        # invoice" says nothing about how the invoice is paid. Both bounded
        # forms above already require payment intent, so a same-episode
        # candidate that shows none is left as an independent finding.
        return None
    # Past the adjacent clause the episode has to be re-established, not
    # merely mentioned: the same transaction, named again, with a payment
    # action performed on it, and nothing in between that names a different
    # one. Without all three the two findings stay independent.
    if (
        _same_payment_episode(anchor.clause, other.clause)
        and _PAYMENT_DIRECTED_RE.search(other.clause)
        and not _intervening_episode(anchor.index, other.index, local)
    ):
        return "REL_SHARED_PAYMENT_ROOT"
    return None


def _intervening_episode(anchor: int, other: int, local: _LocalContext) -> bool:
    """True when a clause between the two names a different transaction."""
    low, high = sorted((anchor, other))
    return any(
        _distinct_episode(local.clause(anchor), local.clause(index))
        for index in range(low + 1, high)
    )


#: Relationships that identify their anchor structurally. Nothing about them
#: can be ambiguous: the anchor is the clause the signal sits in, or the
#: clause that opened the group it sits in.
_UNAMBIGUOUS_RELATIONSHIPS = frozenset({"REL_SAME_CLAUSE", "REL_ANCHORED_LIST"})


def _bound_to_payment(
    anchor: _Signal, other: _Signal, local: _LocalContext, text: str
) -> bool:
    """True when some occurrence of ``other`` binds to some payment anchor.

    Where more than one payment anchor is in range of the same signal and
    those anchors name different transactions, the grammar cannot say which
    one the signal is about -- so it says nothing. Both findings survive
    independently and no combination is formed.
    """
    for other_occurrence in other.occurrences:
        reached: list[_Occurrence] = []
        structural = False
        for anchor_occurrence in anchor.occurrences:
            relationship = _relationship(anchor_occurrence, other_occurrence, local, text)
            if relationship is None:
                continue
            if relationship in _UNAMBIGUOUS_RELATIONSHIPS:
                structural = True
                break
            reached.append(anchor_occurrence)
        if structural:
            return True
        if not reached:
            continue
        candidates = {occurrence.index: occurrence.clause for occurrence in reached}
        if len(candidates) > 1 and any(
            _distinct_episode(left, right)
            for left in candidates.values()
            for right in candidates.values()
        ):
            continue  # competing anchors; do not guess which one it modifies
        return True
    return False


def _evaluate_payment(norm: NormalizedText, local: _LocalContext) -> list[RuleHit]:
    text = norm.security
    hits: list[RuleHit] = []

    demand = _collect_signal(_PAYMENT_DEMAND, text, local)
    off_channel = _collect_signal(_OFF_CHANNEL, text, local)
    threat = _collect_signal(_THREAT, text, local)
    secrecy = _collect_signal(_SECRECY, text, local)
    anti_verification = _collect_signal(_ANTI_VERIFICATION, text, local)
    # The advance-fee rule is itself a cross-clause relationship the design
    # already approves -- "you have won a refund; a clearance fee applies to
    # release it" is one construction spread over two clauses -- so it keeps
    # its whole-text scan.
    advance_fee = _collect_signal(_ADVANCE_FEE, text, local, confined=False)
    ordinary = _is_ordinary(demand)

    def severity_for(signal: _Signal, base: Severity) -> Severity:
        # Framed *everywhere* it appears is discussion; framed somewhere and
        # plain somewhere else is still a demand.
        return base if signal.unframed else Severity.WEAK

    if demand:
        discussion = not demand.unframed
        if ordinary and not (threat or off_channel or secrecy or anti_verification):
            # A bill that says "due now" is a bill.
            hits.append(
                RuleHit(
                    FindingCategory.FINANCIAL_REQUEST,
                    Severity.WEAK,
                    "PAY_ROUTINE_DEMAND",
                    DiagnosticCode.PAYMENT_DEMAND,
                    ambiguous=True,
                )
            )
        else:
            hits.append(
                RuleHit(
                    FindingCategory.COERCIVE_PAYMENT,
                    severity_for(demand, Severity.SUSPICIOUS),
                    "PAY_URGENT_DEMAND",
                    DiagnosticCode.PAYMENT_DEMAND,
                    ambiguous=discussion,
                )
            )

    # Both combinations name a payment, so both must reach one. Without a
    # bound anchor the two families stay separate observations rather than
    # becoming one coercion finding.
    if off_channel and _bound_to_payment(demand, off_channel, local, text):
        hits.append(
            RuleHit(
                FindingCategory.COERCIVE_PAYMENT,
                severity_for(off_channel, Severity.SUSPICIOUS),
                "PAY_OFF_CHANNEL",
                DiagnosticCode.OFF_CHANNEL_PAYMENT,
                ambiguous=not off_channel.unframed,
            )
        )

    if threat and _bound_to_payment(demand, threat, local, text):
        hits.append(
            RuleHit(
                FindingCategory.COERCIVE_PAYMENT,
                severity_for(threat, Severity.SUSPICIOUS),
                "PAY_THREAT_COMBINATION",
                DiagnosticCode.CONSEQUENCE_THREAT,
                ambiguous=not threat.unframed,
            )
        )

    if advance_fee and not _is_ordinary(advance_fee):
        hits.append(
            RuleHit(
                FindingCategory.ADVANCE_FEE,
                severity_for(advance_fee, Severity.SUSPICIOUS),
                "PAY_ADVANCE_FEE",
                DiagnosticCode.ADVANCE_FEE_OFFER,
                ambiguous=not advance_fee.unframed,
            )
        )

    if secrecy:
        hits.append(
            RuleHit(
                FindingCategory.MALICIOUS_URGENCY,
                # Secrecy is only *coercive* when it is secrecy about this
                # payment. An unrelated confidential remark beside an
                # ordinary invoice stays weak.
                severity_for(secrecy,
                             Severity.SUSPICIOUS
                             if _bound_to_payment(demand, secrecy, local, text)
                             else Severity.WEAK),
                "URG_SECRECY",
                DiagnosticCode.SECRECY_PRESSURE,
                ambiguous=not secrecy.unframed,
            )
        )

    if anti_verification:
        hits.append(
            RuleHit(
                FindingCategory.MALICIOUS_URGENCY,
                severity_for(anti_verification, Severity.SUSPICIOUS),
                "URG_ANTI_VERIFICATION",
                DiagnosticCode.ANTI_VERIFICATION_PRESSURE,
                ambiguous=not anti_verification.unframed,
            )
        )
    return hits


# ---------------------------------------------------------------------------
# Urgency (separate from any conclusion).
# ---------------------------------------------------------------------------

_TIME_PRESSURE = _compile_all(
    r"\bwithin (?:the next )?(?:\d+|a few|24|48|72) (?:minutes|hours|hrs)\b",
    r"\b(?:act|respond|reply|pay|click) (?:now|immediately|at once|right away)\b",
    r"\b(?:expires?|expiring|deadline)\b[^.]{0,20}\b(?:today|now|in \d+ (?:minutes|hours))\b",
    r"\blast chance\b", r"\bimmediate action (?:is )?required\b",
)
_ORDINARY_DEADLINE = _compile_all(
    r"\bdue (?:on|by)\b", r"\bby (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\bnext week\b", r"\bend of (?:the )?(?:month|quarter|week)\b",
)


def _evaluate_urgency(norm: NormalizedText, local: _LocalContext) -> list[RuleHit]:
    """Manufactured time pressure, judged at every occurrence.

    Both suppressors are clause-local on purpose. A booking notice stating
    its own expiry, or an invoice carrying an ordinary due date, excuses the
    clause it appears in -- not a countdown three sentences later. Taking
    the first match and letting a due date anywhere in the message cancel it
    is how "due on Friday. Act now or lose access forever" came out clean.
    """
    text = norm.security
    threatening = bool(_collect_signal(_THREAT, text, local))

    for pattern in _TIME_PRESSURE:
        for match in pattern.finditer(text):
            ctx = local.at(match.start())
            if ctx.is_advisory:
                continue
            if ctx.notification and not threatening:
                # A code-delivery or booking notice saying when it expires is
                # stating a fact about itself, not applying pressure.
                continue
            if not threatening and _any(_ORDINARY_DEADLINE, local.clause_at(match.start())):
                continue
            # Never severe on its own: real emergencies are urgent too.
            return [
                RuleHit(
                    FindingCategory.MALICIOUS_URGENCY,
                    Severity.SUSPICIOUS if threatening else Severity.WEAK,
                    "URG_TIME_PRESSURE",
                    DiagnosticCode.ARTIFICIAL_TIME_PRESSURE,
                    ambiguous=True,
                )
            ]
    return []


# ---------------------------------------------------------------------------
# Identity claims.
# ---------------------------------------------------------------------------

_IDENTITY_CLAIM = _compile_all(
    r"\bthis is (?:an )?official\b", r"\bofficial (?:notice|message|communication) from\b",
    r"\bi am (?:from|with|calling from)\b[^.]{0,30}\b(?:bank|support|team|department|service|government|tax)\b",
    r"\b(?:verified|authorised|authorized|certified) (?:sender|account|representative|agent)\b",
    r"\bon behalf of\b[^.]{0,25}\b(?:bank|government|tax|police|court)\b",
)


def _evaluate_identity(norm: NormalizedText, local: _LocalContext, has_asserted_trust: bool) -> list[RuleHit]:
    """An unsupported claim of official identity, sought at every occurrence.

    Stopping at the first claim let a quoted or educational mention of one
    stand in for a real assertion later in the same message. Repeated claims
    are one conflict, so a single finding is still emitted -- the change is
    which occurrences are allowed to decide.
    """
    if has_asserted_trust:
        # A claim matching an independently recorded trust signal is not a
        # contradiction. It is still not *proof* -- the trust signal is,
        # and the claim adds nothing.
        return []
    for pattern in _IDENTITY_CLAIM:
        for match in pattern.finditer(norm.security):
            ctx = local.at(match.start())
            if ctx.is_advisory or ctx.is_discussion:
                continue
            return [
                RuleHit(
                    FindingCategory.IMPERSONATION,
                    Severity.SUSPICIOUS,
                    "ID_UNSUPPORTED_CLAIM",
                    DiagnosticCode.IDENTITY_CLAIM_UNSUPPORTED,
                )
            ]
    return []


# ---------------------------------------------------------------------------
# Links. Parsed locally; never resolved, fetched, or looked up.
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"'\)]+", re.IGNORECASE)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]{1,120})\]\((https?://[^\s\)]{1,300})\)", re.IGNORECASE)
_HOSTLIKE_RE = re.compile(r"\b([a-z0-9-]{2,}(?:\.[a-z0-9-]{2,}){1,3})\b", re.IGNORECASE)
_DECEPTIVE_LABEL_RE = re.compile(
    r"\b(?:secure|login|signin|sign-in|verify|verification|account|update|confirm|support|billing|wallet|recovery)\b",
    re.IGNORECASE,
)


def _host_of(raw: str) -> str | None:
    candidate = raw if "://" in raw else f"http://{raw}"
    try:
        parsed = urllib.parse.urlsplit(candidate)
        return (parsed.hostname or "").lower() or None
    except ValueError:
        return None


def _netloc_of(raw: str) -> str:
    candidate = raw if "://" in raw else f"http://{raw}"
    try:
        return urllib.parse.urlsplit(candidate).netloc.lower()
    except ValueError:
        return ""


def _is_raw_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _registrable_label(host: str) -> str:
    parts = host.split(".")
    return parts[-2] if len(parts) >= 2 else host


def _evaluate_links(norm: NormalizedText, local: _LocalContext) -> list[RuleHit]:
    text = norm.security
    hits: list[RuleHit] = []
    seen_rules: set[str] = set()
    position = {"at": 0}

    def add(rule_id: str, severity: Severity, diagnostic: DiagnosticCode) -> None:
        if rule_id in seen_rules:
            return
        seen_rules.add(rule_id)
        ctx = local.at(position["at"])
        hits.append(
            RuleHit(
                FindingCategory.DECEPTIVE_LINK,
                Severity.WEAK if ctx.is_discussion else severity,
                rule_id,
                diagnostic,
                ambiguous=ctx.is_discussion,
            )
        )

    for match in _URL_RE.finditer(text):
        position["at"] = match.start()
        raw = match.group(0).rstrip(".,;:!?")
        host = _host_of(raw)
        if not host:
            continue
        netloc = _netloc_of(raw)

        if "@" in netloc:
            # Everything before "@" is userinfo, not the destination -- a
            # classic way to make a link read as one site and go to another.
            add("LNK_USERINFO", Severity.SUSPICIOUS, DiagnosticCode.EMBEDDED_CREDENTIALS_IN_URL)
        if _is_raw_ip(host):
            add("LNK_RAW_IP", Severity.SUSPICIOUS, DiagnosticCode.RAW_IP_HOST)
        if "xn--" in host:
            add("LNK_PUNYCODE", Severity.SUSPICIOUS, DiagnosticCode.PUNYCODE_HOST)

        label = _registrable_label(host)
        if label.count("-") >= 2 and _DECEPTIVE_LABEL_RE.search(label):
            add("LNK_DECEPTIVE_LABEL", Severity.SUSPICIOUS, DiagnosticCode.LOOKALIKE_HOST)
        elif host.count(".") >= 3 and _DECEPTIVE_LABEL_RE.search(host.split(".")[0]):
            # A service-sounding word buried in a subdomain while the real
            # registrable domain is something else entirely.
            add("LNK_SUBDOMAIN_DISGUISE", Severity.SUSPICIOUS, DiagnosticCode.LOOKALIKE_HOST)

    for label_match in _MARKDOWN_LINK_RE.finditer(text):
        position["at"] = label_match.start()
        label_text, target = label_match.group(1), label_match.group(2)
        target_host = _host_of(target)
        if not target_host:
            continue
        for candidate in _HOSTLIKE_RE.findall(label_text):
            candidate_host = candidate.lower()
            if "." in candidate_host and candidate_host != target_host:
                if not target_host.endswith("." + candidate_host):
                    add(
                        "LNK_DISPLAY_MISMATCH",
                        Severity.SUSPICIOUS,
                        DiagnosticCode.DISPLAY_TARGET_MISMATCH,
                    )
                break
    return hits


# ---------------------------------------------------------------------------
# Narrowly-scoped unsafe advice.
# ---------------------------------------------------------------------------

_PROFESSIONAL_GUIDANCE = _compile_all(
    r"\b(?:my|your|the) (?:doctor|gp|physician|consultant|pharmacist|nurse|clinician|"
    r"prescriber|doctors|physicians|consultants|pharmacists|nurses|clinicians|prescribers)\b"
    r"[^.]{0,30}\b(?:said|says|advised|told|recommended|prescribed)\b",
    r"\bas (?:prescribed|directed|advised) by\b",
    r"\bunder (?:medical|professional) supervision\b",
    r"\bfollow(?:ing)? (?:your|the) (?:doctor|gp|prescription)\b",
)

#: Clause boundary. Negation, guidance and directive markers are all read
#: *within* one clause, because a global "the text contains 'not' somewhere"
#: exemption is exactly how a warning in one sentence ends up cancelling a
#: dangerous instruction in the next.
#: Clause text, from the one shared segmentation used by the context
#: resolver -- so a colon, a line break, or a bounded weak separator splits
#: an instruction from whatever framed the sentence.

# A negator standing *before* a harmful action turns the clause into a
# warning against it. Position matters: "don't stop your medication" is
# advice not to, while "stop your medication and don't tell anyone" is not.
_ADVICE_NEGATOR = _compile_all(
    r"\b(?:never|do ?n[o']?t|dont|should ?n[o']?t|shouldnt|must ?n[o']?t|mustnt|avoid|no ?one should)\b",
    r"\bit(?:'s| is) (?:dangerous|unsafe|risky|not safe|a bad idea) to\b",
)

# Seeking professional guidance *before* the action is the opposite of
# advising the action. "Ask your doctor before you stop" is safe; "stop
# without asking your doctor" is not, and the qualifier trails the verb
# there, so a position-aware check separates them.
_SEEK_GUIDANCE_FIRST = _compile_all(
    r"\b(?:ask|consult|speak (?:to|with)|talk to|check with|contact|call|see)\b[^.]{0,25}"
    r"\b(?:doctor|gp|physician|clinician|pharmacist|nurse|specialist|consultant|"
    r"prescriber|doctors|physicians|clinicians|pharmacists|nurses|specialists|"
    r"consultants|prescribers|qualified medical professionals?|healthcare professionals?)\b",
    r"\b(?:professional|medical) (?:guidance|advice|supervision)\b",
)

# Two rules below match text that *is* a negation ("don't call an
# ambulance"). Their warning form is therefore a double negation or an
# obligation, which is what this recognizes.
_ENCOURAGE_CARE = _compile_all(
    r"\b(?:never|do ?n[o']?t|dont)\b[^.]{0,25}"
    r"\b(?:hesitate|delay|forget|wait|put off|be afraid|be embarrassed|fail)\b",
    r"\b(?:always|be sure to|make sure|remember to)\b[^.]{0,25}"
    r"\b(?:call|contact|tell|inform|see|visit|seek)\b",
    r"\bif in doubt\b",
)

# Discussing, quoting, or criticizing dangerous advice is not giving it.
_ADVICE_CRITICAL_CONTEXT = _compile_all(
    r"\b(?:if|when|whenever|suppose|imagine)\b[^.]{0,45}"
    r"\b(?:tells? you|asks? you|says? you should|advises? you)\b",
    r"\b(?:that|this|it)(?:'s| is| was) (?:dangerous|unsafe|harmful|bad advice|a myth|misinformation)\b",
    r"\b(?:never|do ?n[o']?t|dont)\b[^.]{0,20}\b(?:follow|listen to|act on|believe)\b",
    r"\b(?:bad|dangerous|unsafe|harmful) advice\b",
    r"\b(?:someone|somebody|a friend|a stranger|a caller) (?:told|messaged|texted|said|sent)\b",
)

# Severe requires the clause to actually direct the reader to act, rather
# than merely mentioning the action.
_ADVICE_DIRECTIVE = _compile_all(
    r"\byou (?:should|shall|must|need to|have to|ought to|can|could|might want to)\b",
    r"\b(?:just|simply|honestly|seriously)\b",
    r"\bi (?:recommend|suggest|advise)\b",
    r"\b(?:make sure|be sure) (?:to|you)\b",
    r"\bit(?:'s| is) (?:fine|ok|okay|safe|better|best) to\b",
    r"\byou(?:'ll| will) be fine\b",
    r"\bno need to\b",
    r"\btrust me\b",
    # "do X instead of <seeking help>" is a substitution instruction even
    # when the clause opens with a bare imperative, so the harmful phrase
    # sits mid-clause and the imperative test above cannot see it. Bounded
    # to acts of seeking help; past-tense narration is already excluded
    # because the harmful verbs are token-bounded.
    r"\binstead of\b[^.]{0,25}"
    r"\b(?:calling|going|seeing|seeking|contacting|visiting|attending|ringing|dialling|dialing)\b",
)

#: A bounded list marker. A line break already ends a clause, so a bullet
#: or an item number is decoration on the front of the next one -- but the
#: imperative test below reads the text *before* the harmful verb, and an
#: unrecognised "- " there is enough to demote a plain instruction. Bounded
#: to marker characters and one- or two-digit item numbers, and requiring
#: trailing space, so "-5 degrees" and "3.5 ml" are untouched.
_LIST_MARKER = r"(?:[-*•–—·]+|\(?\d{1,2}[.)])"

#: A clause whose harmful verb sits at the front is an imperative.
_IMPERATIVE_PREFIX_RE = _c(
    # An opening quotation mark is punctuation, not framing. Refusing to look
    # past it means "stop taking your medications tonight" in quotes reads as
    # a mention rather than an instruction, which hands out an exemption for
    # the price of two characters. Whether the quote is *protected* is a
    # separate question, decided by attribution.
    r"^\s*[\"'“‘(\[]?\s*(?:" + _LIST_MARKER + r"\s+)?"
    + r"(?:\s*(?:and|then|also|so|but|just|please|ok|okay|hey|,)\s*)*$"
)

#: ``True`` when the pattern itself contains the harmful negation, so a
#: leading negator is the *harmful* form rather than the warning form.
_HARM_IS_NEGATION = True
_HARM_IS_INSTRUCTION = False

# Bounded object vocabularies, shared by the rules below so singular and
# plural forms cannot drift apart. Each is an explicit alternation with one
# grouped optional plural suffix -- not stemming, not fuzzy matching, and
# not a medical dictionary. A trailing ``\b`` in the rule keeps every
# alternative token-bounded, so none of these matches inside a longer word.
#
# ``gp`` deliberately carries no plural suffix: "gps" is a navigation
# device far more often than it is two general practitioners.
_MEDICATION_OBJECT = (
    r"(?:medication|medicine|tablet|pill|prescription|prescribed drug|"
    r"prescribed treatment|insulin|antibiotic|painkiller|inhaler|treatment)s?"
)
_DOSE_OBJECT = r"(?:dose|dosage|tablet|pill)s?"
_TREATMENT_OBJECT = r"(?:" + _MEDICATION_OBJECT + r"|(?:dose|dosage)s?)"

#: Consumption or administration of a treatment -- the action a dangerous
#: cessation instruction actually targets.
_CONSUMPTION_HEAD = r"(?:taking|using|administering|consuming|following)"
_CESSATION_VERB = r"(?:stop|quit|discontinue|cease|throw away|bin|come off|give up)"
_CARE_PROVIDER = (
    r"(?:(?:doctor|physician|clinician|prescriber|pharmacist|nurse|paramedic|"
    r"qualified medical professional|healthcare professional|health care professional|"
    r"medical professional)s?|gp)"
)
# Reaching for help, not merely naming a place. Requiring the verb is what
# keeps "don't park near the emergency room" from reading as a refusal of
# care: the terms below are only dangerous as the object of seeking help.
_SEEK_CARE_VERB = (
    r"(?:call|calling|contact|contacting|ring|ringing|dial|dialling|dialing|"
    r"go to|going to|visit|visiting|attend|attending|see|seeing|seek|seeking|"
    r"get|getting|use|using)"
)
_EMERGENCY_CARE_OBJECT = (
    r"(?:(?:an? |the )?ambulances?|(?:the )?ambulance crew|(?:the )?hospital|"
    r"(?:the )?emergency (?:care|services?|department|rooms?)|(?:the )?a ?& ?e|"
    r"urgent medical help|immediate medical attention|9 ?1 ?1|9 ?9 ?9|1 ?1 ?2|"
    r"(?:a |the )?" + _CARE_PROVIDER + r"|help)"
)
_CONCEALMENT_TARGET = (
    r"(?:" + _CARE_PROVIDER + r"|(?:the )?hospital|(?:the )?emergency services?|"
    r"(?:the )?ambulance crew)"
)
_EMERGENCY_SYMPTOM = (
    r"(?:chest pain|difficulty breathing|trouble breathing|shortness of breath|"
    r"heavy bleeding|seizure|stroke|overdose|numbness|slurred speech|"
    r"emergency symptom|symptom|dangerous condition)s?"
)

_UNSAFE_ADVICE_RULES = (
    (
        "ADV_STOP_MEDICATION",
        # A. the verb governs a consumption or administration action, which
        #    in turn governs the treatment: "stop taking your medication".
        r"\b" + _CESSATION_VERB + r"\b" + _NP_PREFIX + r"\s+" + _CONSUMPTION_HEAD + r"\b"
        + _NP_PREFIX + r"\s+" + _TREATMENT_OBJECT + r"\b" + _NP_HEAD_END
        # B. the verb governs the treatment directly, and the treatment word
        #    is the head of that phrase: "discontinue the prescribed medicine".
        #    First-person possessives are excluded here -- see
        #    ``_NP_PREFIX_DIRECTED``.
        + r"|\b" + _CESSATION_VERB + r"\b" + _NP_PREFIX_DIRECTED + r"\s+"
        + _TREATMENT_OBJECT + r"\b" + _NP_HEAD_END,
        DiagnosticCode.MEDICATION_DISCONTINUATION,
        _HARM_IS_INSTRUCTION,
    ),
    (
        "ADV_DANGEROUS_DOSE",
        # The dose must be the head of what the directive governs, for the
        # same reason: "change the dosage reminder" is an administrative
        # request, not a dose change.
        r"\b(?:take|swallow|use)\b[^.]{0,25}\b(?:double|triple|twice|three times|extra|more than)\b"
        r"[^.]{0,25}\b" + _DOSE_OBJECT + r"\b" + _NP_HEAD_END
        + r"|\btake\b[^.]{0,15}\b(?:\d{2,})\b[^.]{0,15}\b(?:tablet|pill)s?\b" + _NP_HEAD_END
        + r"|\b(?:double|triple|halve|change|alter|increase|reduce)\b" + _NP_PREFIX
        + r"\s+(?:dose|dosage)s?\b" + _NP_HEAD_END,
        DiagnosticCode.DANGEROUS_DOSAGE,
        _HARM_IS_INSTRUCTION,
    ),
    (
        "ADV_HARMFUL_INGESTION",
        r"\b(?:drink|swallow|ingest|inhale|gargle)\b[^.]{0,25}"
        r"\b(?:bleach|disinfectant|antifreeze|petrol|gasoline|turpentine|"
        r"hydrogen peroxide|cleaning product)s?\b",
        DiagnosticCode.HARMFUL_INGESTION,
        _HARM_IS_INSTRUCTION,
    ),
    (
        "ADV_IGNORE_EMERGENCY_SYMPTOMS",
        r"\b(?:ignore|dismiss|push through|sleep off|walk off|shake off)\b[^.]{0,30}"
        r"\b" + _EMERGENCY_SYMPTOM + r"\b",
        DiagnosticCode.EMERGENCY_SYMPTOMS_IGNORED,
        _HARM_IS_INSTRUCTION,
    ),
    (
        "ADV_AVOID_EMERGENCY_CARE",
        r"\b(?:do ?n[o']?t|dont|no need to|avoid)\b[^.]{0,25}"
        r"\b" + _SEEK_CARE_VERB + r"\b[^.]{0,15}\b" + _EMERGENCY_CARE_OBJECT + r"\b"
        r"|\binstead of\b[^.]{0,20}\b" + _SEEK_CARE_VERB + r"\b[^.]{0,15}"
        r"\b" + _EMERGENCY_CARE_OBJECT + r"\b",
        DiagnosticCode.EMERGENCY_CARE_DISCOURAGED,
        _HARM_IS_NEGATION,
    ),
    (
        "ADV_CONCEAL_FROM_CARE",
        r"\b(?:do ?n[o']?t|dont|never)\b[^.]{0,25}"
        r"\b(?:tell|mention|inform|show|report(?: to)?|disclose(?: to)?)\b[^.]{0,25}"
        r"\b" + _CONCEALMENT_TARGET + r"\b",
        DiagnosticCode.CONCEALMENT_FROM_CARE,
        _HARM_IS_NEGATION,
    ),
)
_UNSAFE_COMPILED = tuple(
    (rid, _c(pat), diag, negated) for rid, pat, diag, negated in _UNSAFE_ADVICE_RULES
)


def _clauses(text: str, boundaries: tuple[int, ...] = ()) -> tuple[str, ...]:
    return tuple(
        part for part in (text[a:b].strip() for a, b in _segment(text, boundaries)) if part
    )


def _advice_is_warning(clause: str, start: int, harm_is_negation: bool) -> bool:
    """True when this clause argues *against* the action it names."""
    if harm_is_negation:
        # The negation is inside the match, so only a double negation or an
        # explicit obligation to seek care can flip it back to a warning.
        return _any(_ENCOURAGE_CARE, clause)
    prefix = clause[:start]
    return _any(_ADVICE_NEGATOR, prefix) or _any(_SEEK_GUIDANCE_FIRST, prefix)


def _local_prefix(clause: str, start: int) -> str:
    """The text since the last accepted separator: this clause's own offset 0.

    Actionability is a property of the clause the instruction sits in, not of
    the paragraph it arrived with. Measuring the imperative against the front
    of the whole document means one unremarkable introductory clause --
    "lunch is at one, but stop taking your medications tonight" -- moves the
    verb away from position zero and quietly demotes a dangerous instruction.
    Cutting at the last accepted separator restores the local offset without
    re-segmenting anything or widening what counts as a boundary.
    """
    cut = 0
    for match in _CLAUSE_BOUNDARY_RE.finditer(clause, 0, start):
        cut = match.end()
    for match in _WEAK_SEPARATOR_RE.finditer(clause, 0, start):
        cut = max(cut, match.end())
    return clause[cut:start]


def _advice_is_directive(clause: str, start: int) -> bool:
    """True when the clause tells the reader to do it, not merely names it."""
    return _any(_ADVICE_DIRECTIVE, clause) or bool(
        _IMPERATIVE_PREFIX_RE.match(_local_prefix(clause, start))
    )


def _evaluate_unsafe_advice(norm: NormalizedText, local: _LocalContext) -> list[RuleHit]:
    """Narrow, clause-local dangerous-advice detection.

    Severity is decided here and nowhere else. A clear, actionable,
    unambiguous instruction toward substantial physical harm is severe and
    therefore obliges the ordinary hard veto; everything softer -- a
    mention, a clinician-guided statement, a hypothetical -- stays below it.
    No diagnostic, rule identifier, or indicator carries any of the text.

    The *advisory* flag is deliberately not consulted. It recognizes
    credential warnings, which say nothing about medical advice, and
    honouring it here would make "never share your password with anyone" a
    prefix that silences whatever follows it. Educational and quoted framing
    is read per clause for the same reason, and a document-level discussion
    tone caps severity rather than exempting a clause outright.
    """
    hits: list[RuleHit] = []
    for span_start, span_end in _segment(norm.security, norm.boundary_offsets):
        clause = norm.security[span_start:span_end].strip()
        if not clause:
            continue
        # A clause that *is* an attributed quotation is reported speech, and
        # the quotation marks alone no longer say so -- attribution does.
        # Read where the clause actually sits, so "someone sent me a message
        # saying: <danger>" stays protected while a bare quoted demand does
        # not. The advisory flag is still not consulted, for the reason above.
        quoted_here = local.at(span_start)
        if quoted_here.quoted or quoted_here.educational:
            continue
        if (
            _any(_ADVICE_CRITICAL_CONTEXT, clause)
            or _any(_EDUCATIONAL_PATTERNS, clause)
            or _any(_QUOTED_PATTERNS, clause)
        ):
            # Quoting, supposing, teaching, or condemning it -- not giving it.
            continue
        guided = _any(_PROFESSIONAL_GUIDANCE, clause)
        for rule_id, pattern, diagnostic, harm_is_negation in _UNSAFE_COMPILED:
            match = pattern.search(clause)
            if match is None:
                continue
            if _advice_is_warning(clause, match.start(), harm_is_negation):
                continue
            if guided:
                # Relaying what a clinician actually said is not unsafe advice.
                hits.append(
                    RuleHit(
                        FindingCategory.UNSAFE_ADVICE,
                        Severity.WEAK,
                        rule_id,
                        diagnostic,
                        ambiguous=True,
                    )
                )
                continue
            if _advice_is_directive(clause, match.start()):
                hits.append(
                    RuleHit(FindingCategory.UNSAFE_ADVICE, Severity.SEVERE, rule_id, diagnostic)
                )
                continue
            hits.append(
                RuleHit(
                    FindingCategory.UNSAFE_ADVICE,
                    Severity.SUSPICIOUS,
                    rule_id,
                    diagnostic,
                    ambiguous=False,
                )
            )
    return _deduplicate(hits)


def _deduplicate(hits: list[RuleHit]) -> list[RuleHit]:
    """Keep the strongest hit per rule; clauses must not multiply findings."""
    strongest: dict[str, RuleHit] = {}
    for hit in hits:
        current = strongest.get(hit.rule_id)
        if current is None or hit.severity.rank > current.severity.rank:
            strongest[hit.rule_id] = hit
    return [strongest[rule_id] for rule_id in sorted(strongest)]


# ---------------------------------------------------------------------------
# Propagation.
# ---------------------------------------------------------------------------

_CHAIN_LANGUAGE = _compile_all(
    r"\bforward this\b", r"\bsend this to\b[^.]{0,20}\b(?:\d+|everyone|all your|friends|contacts|groups)\b",
    r"\bshare (?:this )?with (?:everyone|all your|\d+)\b", r"\bcopy and paste this\b",
    r"\bpass (?:this|it) on\b", r"\bdo ?n[o']?t break the chain\b",
)
_RESEND_PRESSURE = _compile_all(
    r"\bresend\b", r"\bshare (?:this )?immediately\b", r"\bspread the word\b",
    r"\bbefore (?:it(?:'| i)?s )?(?:deleted|taken down|removed)\b",
)
_MASS_WARNING = _compile_all(
    r"\bwarning to all\b", r"\bthis is not a hoax\b", r"\bconfirmed by\b[^.]{0,30}\b(?:police|government|ministry|who)\b",
    r"\bshare to save (?:lives|a life)\b", r"\bplease alert everyone\b",
)

#: Above this, forwarding metadata is worth noting. Chosen as a plain
#: threshold rather than a tuned value; it is a hint, not a verdict.
HIGH_FORWARD_THRESHOLD = 5


def evaluate_propagation(norm: NormalizedText | None, forwarding_count: int | None) -> list[RuleHit]:
    """Propagation indicators. Never conclusive, never severe."""
    hits: list[RuleHit] = []
    if forwarding_count is not None and forwarding_count >= HIGH_FORWARD_THRESHOLD:
        hits.append(
            RuleHit(
                FindingCategory.PROPAGATION_RISK,
                Severity.WEAK,
                "PROP_HIGH_FORWARD_COUNT",
                DiagnosticCode.HIGH_FORWARD_COUNT,
                ambiguous=True,
            )
        )
    if norm is None:
        return hits
    text = norm.security
    for patterns, rule_id, diagnostic in (
        (_CHAIN_LANGUAGE, "PROP_CHAIN_LANGUAGE", DiagnosticCode.CHAIN_MESSAGE_LANGUAGE),
        (_RESEND_PRESSURE, "PROP_RESEND_PRESSURE", DiagnosticCode.RESEND_PRESSURE),
        (_MASS_WARNING, "PROP_MASS_WARNING", DiagnosticCode.UNVERIFIABLE_MASS_WARNING),
    ):
        if _any(patterns, text):
            hits.append(
                RuleHit(
                    FindingCategory.PROPAGATION_RISK,
                    Severity.WEAK,
                    rule_id,
                    diagnostic,
                    ambiguous=True,
                )
            )
    return hits


# ---------------------------------------------------------------------------
# Entry point for one piece of content.
# ---------------------------------------------------------------------------


def evaluate_content(norm: NormalizedText, has_asserted_trust: bool = False) -> list[RuleHit]:
    """Run every content rule family over one normalized piece of content."""
    local = _LocalContext(norm.security, norm.boundary_offsets)
    hits: list[RuleHit] = []
    hits.extend(_evaluate_injection(norm, local))
    hits.extend(_evaluate_credentials(norm, local))
    hits.extend(_evaluate_payment(norm, local))
    hits.extend(_evaluate_urgency(norm, local))
    hits.extend(_evaluate_identity(norm, local, has_asserted_trust))
    hits.extend(_evaluate_links(norm, local))
    hits.extend(_evaluate_unsafe_advice(norm, local))
    return hits
