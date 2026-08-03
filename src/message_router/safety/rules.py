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

_WARNING_PATTERNS = _compile_all(
    # The object matters. "Never share your code" is a security warning;
    # "don't tell your doctor" and "don't tell anyone" are not, and treating
    # them as advisories would silence the unsafe-advice and secrecy rules
    # that exist precisely to catch them.
    r"\b(?:never|do ?n[o']?t|dont|don t)\b[^.]{0,40}"
    # The transmission actions belong here for the same reason they are a
    # request: "never message support your PIN" is the warning form of the
    # very construction the credential layer treats as a disclosure.
    r"\b(?:share|give|send|disclose|reveal|tell|enter|message|dm|text|whatsapp)\b[^.]{0,30}"
    r"\b(?:otp|pins?|personal identification numbers?|password|passcode|code|codes|"
    r"card|cvv|credentials|login|details|account details|security answer)\b",
    r"\bwe (?:will )?never (?:ask|request|call|email|text)\b",
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
    return bool(_MESSAGE_TRANSMISSION_RE.search(stripped))


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
#: A clause that is itself a quotation, and the framing that introduces one.
_QUOTE_ENCLOSED_RE = _c(r"^[\"'“‘(\[]|[\"'”’)\]]$")
_QUOTE_INTRODUCER_RE = _c(
    r"\b(?:said|says|saying|reads|wrote|writes|asked|asks|claims|claimed|"
    r"message|text|sms|email|note|post)\b[^.]{0,20}$"
)


class _LocalContext:
    """Framing resolved per clause, with one narrow inheritance rule.

    A clause carries its own warning, notification, educational and quoted
    flags. The single exception is a clause that *is* a quotation — enclosed
    in quotes, or introduced by a preceding clause that announces one — which
    inherits the quoted and educational framing of the clause that framed it.
    That is the "narrowly associated clause" case, and it is the only way
    framing crosses a boundary.
    """

    __slots__ = ("_spans", "_flags", "_fallback", "_quoted")

    def __init__(self, text: str, boundaries: tuple[int, ...] = ()) -> None:
        self._spans = [
            (start, end, text[start:end]) for start, end in _segment(text, boundaries)
        ]
        self._fallback = _flags_for(text)
        self._quoted = _quoted_spans(text)

        raw = [_flags_for(clause) for _, _, clause in self._spans]
        enclosed = _enclosing_quotation(tuple(clause for _, _, clause in self._spans))
        resolved: list[ContextFlags] = []
        for index, flags in enumerate(raw):
            clause = self._spans[index][2].strip()
            # A quotation that runs past a line break is still one quotation.
            # While it stays open, the clauses inside it keep the framing of
            # the clause that opened it.
            if index and (enclosed[index] or _QUOTE_ENCLOSED_RE.search(clause)):
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

_REQUEST_VERB = (
    r"(?:send|share|give|tell|forward|reply with|provide|submit|enter|type|confirm|"
    r"verify|read out|text|paste|hand over|reveal|disclose|transmit)"
)
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


def _evaluate_payment(norm: NormalizedText, local: _LocalContext) -> list[RuleHit]:
    text = norm.security
    hits: list[RuleHit] = []

    def first(patterns):
        for pattern in patterns:
            match = pattern.search(text)
            if match is not None:
                return match
        return None

    demand_at = first(_PAYMENT_DEMAND)
    anchor = demand_at or first(_THREAT) or first(_OFF_CHANNEL) or first(_ADVANCE_FEE)
    ctx = local.at(anchor.start() if anchor else 0)
    if ctx.is_advisory:
        return hits

    demand = _any(_PAYMENT_DEMAND, text)
    off_channel = _any(_OFF_CHANNEL, text)
    threat = _any(_THREAT, text)
    secrecy = _any(_SECRECY, text)
    anti_verification = _any(_ANTI_VERIFICATION, text)
    advance_fee = _any(_ADVANCE_FEE, text)
    ordinary = _any(_ORDINARY_BILLING, text) or ctx.notification
    discussion = ctx.is_discussion

    def severity_for(base: Severity) -> Severity:
        if discussion:
            return Severity.WEAK
        return base

    if demand:
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
                    severity_for(Severity.SUSPICIOUS),
                    "PAY_URGENT_DEMAND",
                    DiagnosticCode.PAYMENT_DEMAND,
                    ambiguous=discussion,
                )
            )

    if off_channel and (demand or threat):
        hits.append(
            RuleHit(
                FindingCategory.COERCIVE_PAYMENT,
                severity_for(Severity.SUSPICIOUS),
                "PAY_OFF_CHANNEL",
                DiagnosticCode.OFF_CHANNEL_PAYMENT,
                ambiguous=discussion,
            )
        )

    if threat and (demand or off_channel):
        hits.append(
            RuleHit(
                FindingCategory.COERCIVE_PAYMENT,
                severity_for(Severity.SUSPICIOUS),
                "PAY_THREAT_COMBINATION",
                DiagnosticCode.CONSEQUENCE_THREAT,
                ambiguous=discussion,
            )
        )

    if advance_fee and not ordinary:
        hits.append(
            RuleHit(
                FindingCategory.ADVANCE_FEE,
                severity_for(Severity.SUSPICIOUS),
                "PAY_ADVANCE_FEE",
                DiagnosticCode.ADVANCE_FEE_OFFER,
                ambiguous=discussion,
            )
        )

    if secrecy:
        hits.append(
            RuleHit(
                FindingCategory.MALICIOUS_URGENCY,
                severity_for(Severity.SUSPICIOUS if demand or threat else Severity.WEAK),
                "URG_SECRECY",
                DiagnosticCode.SECRECY_PRESSURE,
                ambiguous=discussion,
            )
        )

    if anti_verification:
        hits.append(
            RuleHit(
                FindingCategory.MALICIOUS_URGENCY,
                severity_for(Severity.SUSPICIOUS),
                "URG_ANTI_VERIFICATION",
                DiagnosticCode.ANTI_VERIFICATION_PRESSURE,
                ambiguous=discussion,
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
    text = norm.security
    pressure = None
    for pattern in _TIME_PRESSURE:
        pressure = pattern.search(text)
        if pressure is not None:
            break
    ctx = local.at(pressure.start() if pressure else 0)
    if ctx.is_advisory or pressure is None:
        return []
    threatening = _any(_THREAT, text)
    if ctx.notification and not threatening:
        # A code-delivery or booking notice saying when it expires is
        # stating a fact about itself, not applying pressure.
        return []
    if _any(_ORDINARY_DEADLINE, text) and not threatening:
        return []
    # Never severe on its own: real emergencies are urgent too.
    return [
        RuleHit(
            FindingCategory.MALICIOUS_URGENCY,
            Severity.WEAK if not _any(_THREAT, text) else Severity.SUSPICIOUS,
            "URG_TIME_PRESSURE",
            DiagnosticCode.ARTIFICIAL_TIME_PRESSURE,
            ambiguous=True,
        )
    ]


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
    claim = None
    for pattern in _IDENTITY_CLAIM:
        claim = pattern.search(norm.security)
        if claim is not None:
            break
    if claim is None:
        return []
    ctx = local.at(claim.start())
    if ctx.is_advisory or ctx.is_discussion:
        return []
    if has_asserted_trust:
        # A claim matching an independently recorded trust signal is not a
        # contradiction. It is still not *proof* -- the trust signal is,
        # and the claim adds nothing.
        return []
    return [
        RuleHit(
            FindingCategory.IMPERSONATION,
            Severity.SUSPICIOUS,
            "ID_UNSUPPORTED_CLAIM",
            DiagnosticCode.IDENTITY_CLAIM_UNSUPPORTED,
        )
    ]


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
    r"^\s*(?:" + _LIST_MARKER + r"\s+)?"
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


def _advice_is_directive(clause: str, start: int) -> bool:
    """True when the clause tells the reader to do it, not merely names it."""
    return _any(_ADVICE_DIRECTIVE, clause) or bool(_IMPERATIVE_PREFIX_RE.match(clause[:start]))


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
    for clause in _clauses(norm.security, norm.boundary_offsets):
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
