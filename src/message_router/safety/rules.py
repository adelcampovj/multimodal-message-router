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
# Context classification.
# ---------------------------------------------------------------------------

_WARNING_PATTERNS = _compile_all(
    # The object matters. "Never share your code" is a security warning;
    # "don't tell your doctor" and "don't tell anyone" are not, and treating
    # them as advisories would silence the unsafe-advice and secrecy rules
    # that exist precisely to catch them.
    r"\b(?:never|do ?n[o']?t|dont|don t)\b[^.]{0,40}"
    r"\b(?:share|give|send|disclose|reveal|tell|enter)\b[^.]{0,30}"
    r"\b(?:otp|pin|password|passcode|code|codes|card|cvv|credentials|"
    r"login|details|account details|security answer)\b",
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
    r"(?:otp|one[\s-]?time[\s-]?(?:password|code|pin)|\bpin\b|passcode|password|"
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
    text = norm.folded
    return ContextFlags(
        warning=_any(_WARNING_PATTERNS, text) or _is_pronoun_warning(text),
        notification=_any(_NOTIFICATION_PATTERNS, text),
        educational=_any(_EDUCATIONAL_PATTERNS, text),
        quoted=_any(_QUOTED_PATTERNS, text),
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


def _evaluate_injection(norm: NormalizedText, ctx: ContextFlags) -> list[RuleHit]:
    hits: list[RuleHit] = []
    for rule_id, pattern, diagnostic in _INJECTION_COMPILED:
        if not pattern.search(norm.folded):
            continue
        if ctx.is_discussion:
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
    r"verify|read out|text|paste|hand over)"
)
_RECIPIENT_POSSESSIVE = r"(?:your|the|me|us|it)"

_SECRET_TARGET = (
    r"(?:otp|one[\s-]?time[\s-]?(?:password|code|pin)|\bpin\b|passcode|password|"
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


def _evaluate_credentials(norm: NormalizedText, ctx: ContextFlags) -> list[RuleHit]:
    text = norm.folded
    hits: list[RuleHit] = []

    for rule_id, request_pattern, target_pattern, diagnostic in _CREDENTIAL_COMPILED:
        target_present = bool(target_pattern.search(text))
        request_present = bool(request_pattern.search(text))

        if not target_present:
            continue

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

    if not hits and not ctx.is_advisory and not ctx.is_discussion and not ctx.notification:
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


def _evaluate_payment(norm: NormalizedText, ctx: ContextFlags) -> list[RuleHit]:
    text = norm.folded
    hits: list[RuleHit] = []
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


def _evaluate_urgency(norm: NormalizedText, ctx: ContextFlags) -> list[RuleHit]:
    text = norm.folded
    if ctx.is_advisory or not _any(_TIME_PRESSURE, text):
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


def _evaluate_identity(norm: NormalizedText, ctx: ContextFlags, has_asserted_trust: bool) -> list[RuleHit]:
    if ctx.is_advisory or ctx.is_discussion:
        return []
    if not _any(_IDENTITY_CLAIM, norm.folded):
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


def _evaluate_links(norm: NormalizedText, ctx: ContextFlags) -> list[RuleHit]:
    text = norm.folded
    hits: list[RuleHit] = []
    seen_rules: set[str] = set()

    def add(rule_id: str, severity: Severity, diagnostic: DiagnosticCode) -> None:
        if rule_id in seen_rules:
            return
        seen_rules.add(rule_id)
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

    for label_text, target in _MARKDOWN_LINK_RE.findall(text):
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
    r"\b(?:my|your|the) (?:doctor|gp|physician|consultant|pharmacist|nurse)\b[^.]{0,30}"
    r"\b(?:said|says|advised|told|recommended|prescribed)\b",
    r"\bas (?:prescribed|directed|advised) by\b",
    r"\bunder (?:medical|professional) supervision\b",
    r"\bfollow(?:ing)? (?:your|the) (?:doctor|gp|prescription)\b",
)

#: Clause boundary. Negation, guidance and directive markers are all read
#: *within* one clause, because a global "the text contains 'not' somewhere"
#: exemption is exactly how a warning in one sentence ends up cancelling a
#: dangerous instruction in the next.
_CLAUSE_SPLIT_RE = re.compile(r"[.!?;\n]+")

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
    r"\b(?:doctor|gp|physician|clinician|pharmacist|nurse|specialist|consultant)\b",
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
)

#: A clause whose harmful verb sits at the front is an imperative.
_IMPERATIVE_PREFIX_RE = _c(r"^(?:\s*(?:and|then|also|so|but|just|please|ok|okay|hey|,)\s*)*$")

#: ``True`` when the pattern itself contains the harmful negation, so a
#: leading negator is the *harmful* form rather than the warning form.
_HARM_IS_NEGATION = True
_HARM_IS_INSTRUCTION = False

_UNSAFE_ADVICE_RULES = (
    (
        "ADV_STOP_MEDICATION",
        r"\b(?:stop|quit|discontinue|throw away|bin|come off)\b[^.]{0,30}"
        r"\b(?:medication|medicine|tablets|pills|prescription|insulin|antibiotics|inhaler|treatment)\b",
        DiagnosticCode.MEDICATION_DISCONTINUATION,
        _HARM_IS_INSTRUCTION,
    ),
    (
        "ADV_DANGEROUS_DOSE",
        r"\b(?:take|swallow|use)\b[^.]{0,25}\b(?:double|triple|twice|three times|extra|more than)\b"
        r"[^.]{0,25}\b(?:dose|doses|tablets|pills)\b"
        r"|\btake\b[^.]{0,15}\b(?:\d{2,})\b[^.]{0,15}\b(?:tablets|pills)\b"
        r"|\b(?:double|triple|halve)\b[^.]{0,20}\b(?:dose|dosage)\b",
        DiagnosticCode.DANGEROUS_DOSAGE,
        _HARM_IS_INSTRUCTION,
    ),
    (
        "ADV_HARMFUL_INGESTION",
        r"\b(?:drink|swallow|ingest|inhale|gargle)\b[^.]{0,25}"
        r"\b(?:bleach|disinfectant|antifreeze|petrol|gasoline|turpentine|hydrogen peroxide)\b",
        DiagnosticCode.HARMFUL_INGESTION,
        _HARM_IS_INSTRUCTION,
    ),
    (
        "ADV_IGNORE_EMERGENCY_SYMPTOMS",
        r"\b(?:ignore|dismiss|push through|sleep off|walk off|shake off)\b[^.]{0,30}"
        r"\b(?:chest pain|difficulty breathing|trouble breathing|shortness of breath|"
        r"heavy bleeding|seizure|stroke|overdose|numbness|slurred speech|symptoms)\b",
        DiagnosticCode.EMERGENCY_SYMPTOMS_IGNORED,
        _HARM_IS_INSTRUCTION,
    ),
    (
        "ADV_AVOID_EMERGENCY_CARE",
        r"\b(?:do ?n[o']?t|dont|no need to|avoid)\b[^.]{0,30}"
        r"\b(?:call an ambulance|go to (?:the )?hospital|emergency room|a ?& ?e|call 9 ?1 ?1|call 9 ?9 ?9|call 1 ?1 ?2|see a doctor|seek help)\b"
        r"|\binstead of\b[^.]{0,25}"
        r"\b(?:calling an ambulance|going to (?:the )?hospital|the emergency room|seeing a doctor|seeking help)\b",
        DiagnosticCode.EMERGENCY_CARE_DISCOURAGED,
        _HARM_IS_NEGATION,
    ),
    (
        "ADV_CONCEAL_FROM_CARE",
        r"\b(?:do ?n[o']?t|dont|never)\b[^.]{0,25}\b(?:tell|mention|inform|show)\b[^.]{0,25}"
        r"\b(?:doctor|gp|hospital|nurse|paramedic|pharmacist)\b",
        DiagnosticCode.CONCEALMENT_FROM_CARE,
        _HARM_IS_NEGATION,
    ),
)
_UNSAFE_COMPILED = tuple(
    (rid, _c(pat), diag, negated) for rid, pat, diag, negated in _UNSAFE_ADVICE_RULES
)


def _clauses(text: str) -> tuple[str, ...]:
    return tuple(part for part in (p.strip() for p in _CLAUSE_SPLIT_RE.split(text)) if part)


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


def _evaluate_unsafe_advice(norm: NormalizedText, ctx: ContextFlags) -> list[RuleHit]:
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
    document_is_discussion = ctx.is_discussion
    hits: list[RuleHit] = []
    for clause in _clauses(norm.folded):
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
            if _advice_is_directive(clause, match.start()) and not document_is_discussion:
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
                    ambiguous=document_is_discussion,
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
    text = norm.folded
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
    ctx = classify_context(norm)
    hits: list[RuleHit] = []
    hits.extend(_evaluate_injection(norm, ctx))
    hits.extend(_evaluate_credentials(norm, ctx))
    hits.extend(_evaluate_payment(norm, ctx))
    hits.extend(_evaluate_urgency(norm, ctx))
    hits.extend(_evaluate_identity(norm, ctx, has_asserted_trust))
    hits.extend(_evaluate_links(norm, ctx))
    hits.extend(_evaluate_unsafe_advice(norm, ctx))
    return hits
