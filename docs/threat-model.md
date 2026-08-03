# Threat Model

Scope note: this file documents the deterministic safety-analysis layer
(`message_router.safety`). Routing, classification, reasons, and confidence
are not implemented yet and are out of scope here.

## Trust boundary

Everything that arrives with a message is untrusted, including content the
system itself extracted from it. Content is data; it is never configuration
and never an instruction.

| Source | Provenance | Trusted? |
|---|---|---|
| Fields from the surrounding application | `structured_metadata` | Structured, **not** verified |
| Text a person typed | `authored_text` | No |
| Text read off an image | `ocr_text` | No |
| Speech converted to text | `asr_transcript` | No |
| Content behind a link | `url_derived` | No |
| Payload decoded from a scannable code | `qr_derived` | No |
| A model's interpretation | `model_interpretation` | No |
| A deterministic check's own conclusion | `derived_signal` | Reviewable, not content |

Two properties hold for *every* provenance without exception, and are
asserted by tests that iterate the whole enum:

* `carries_instruction_authority` is always `False`;
* `confers_verified_trust` is always `False`.

Structured metadata being structured is not the same as being verified.
Trust is only ever positive evidence someone recorded on purpose, as a
`TrustSignal`.

## Finding taxonomy

| Category | What it reports |
|---|---|
| `prompt_injection` | Content trying to change how the system treats it |
| `credential_solicitation` | A request to disclose an authentication secret |
| `financial_request` | An ordinary-looking payment request |
| `coercive_payment` | Payment pressure with threat, urgency, or an off-channel method |
| `advance_fee` | Pay-first-to-receive prize, refund, or reward |
| `impersonation` | An identity claim nothing independently supports |
| `deceptive_link` | Host shapes designed to read as something they are not |
| `malicious_urgency` | Manufactured time pressure, secrecy, anti-verification |
| `unsafe_advice` | Narrow, explicit dangerous instructions |
| `propagation_risk` | Chain-message and mass-forwarding shapes |
| `modality_integrity` | Reserved for source-quality findings |

A finding carries a rule ID, a category, a provenance, a sanitized source
reference, and a fixed diagnostic code. It carries **no** content: no
matched text, no secret, no URL, no payload, no transcript, no path, and no
character spans.

## Severity and veto semantics

| Severity | Meaning | Obligation |
|---|---|---|
| `weak` | Noticed; context for other findings | None |
| `suspicious` | Well-formed concern, not conclusive | None |
| `severe` | Strongly supported | A later stage **must** veto |

`SafetyAssessment.hard_veto_required` is a computed property derived from
the findings, not a stored field, so it cannot drift from the evidence and
cannot be asserted independently of it. This layer records the obligation;
it does not choose the action that discharges it, and it produces no
routing action, message category, rationale, or confidence value.

Severe is reserved for strongly supported cases: credential-disclosure
requests (one-time codes, PINs, passwords, login/recovery/verification
codes, card details, banking access), explicit attempts to override system
behaviour, and coercive payment corroborated by a deceptive link or an
unsupported identity claim. A *mention* is never enough.

## What deliberately stays quiet

The rules are built so honest traffic produces nothing. Four context
classes suppress or downgrade request-shaped rules:

* **warning** — "we will never ask for your code" is advice, not a request.
  Scoped to warnings *about credentials*, so "don't tell anyone" and
  "don't tell your doctor" are not mistaken for security advice.
* **notification** — "your code is …" delivers something. It does **not**
  suppress a genuine request in the same message: "your code is X, now send
  it to me" is still severe.
* **educational** — training material describing attacks.
* **quoted** — a user forwarding something suspicious to ask about it.

Ordinary invoices, receipts, statements, delivery notices, appointment
reminders, and plain deadlines produce no findings. Urgency alone is never
severe, because genuine emergencies are urgent too.

## Modality reliability

Each source reports one state: `not_applicable`, `analyzed`, `empty`,
`unavailable`, `unsupported`, or `failed`. These are kept distinct because
collapsing them loses the difference between "we could not look" and "we
looked and there was nothing".

* Failure, unavailability, and empty extraction are **never** evidence of
  safety; they mark the analysis incomplete.
* A weak or failed source cannot erase a finding from a stronger one — a
  severe finding from an image survives a failed transcription.
* Extracted content never replaces authored content, and a finding's
  reliability is capped by the provenance it came from, so extracted text
  can never outrank what a person actually wrote.
* Sources are analysed separately and never concatenated.

## Isolation from engagement

Safety analysis takes a message, its media analyses, and an optional
forwarding count. It does not accept preferences, engagement history, quiet
hours, silenced-conversation state, or notification tolerance — those are
not parameters of the function, which is enforced by a signature test and
an AST scan over the whole package. A popular sender and an ignored thread
receive identical analysis.

## Known limitations

* **Coverage is finite and English-shaped.** Detection is pattern-based.
  Paraphrase, other languages, and novel phrasings will be missed.
* **No findings is not proof of safety.** It means a finite list did not
  match.
* **Normalization is partial.** Homoglyph and spacing evasion are handled;
  a narrow substitution map is applied only to the squeezed view. Broader
  leetspeak folding is deliberately not applied, because rewriting digits
  into letters invents matches that were never in the text.
* **Squeezed matching is length-gated.** With no word boundaries, short
  targets match inside ordinary words, so they are excluded.
* **Links are parsed, never resolved.** No DNS, no HTTP, no redirect
  following, no reputation lookup. Deception is judged from host shape
  alone, which cannot detect a hostile site at an innocuous address.
* **Lookalike detection is structural.** There is no brand list, so an
  imitation of a specific well-known domain is only caught when its shape
  is itself suspicious.
* **Unsafe advice is deliberately narrow.** It flags a few explicit
  dangerous instructions. It does not diagnose, does not classify medical
  conversation, and emits no health details.
* **Processing is bounded** at 20,000 characters per source; longer content
  is analysed up to the limit and flagged as truncated.

## Explicitly out of scope for this layer

Routing decisions, message classification, reason generation, confidence
calibration, evidence selection or ranking, semantic similarity, model
prompting, and any use of preferences or engagement to soften a finding.
