# Routing Policy

How exactly one action is selected. This document covers the ordered
policy; classification and features are in `architecture.md`, and risk
findings are in `threat-model.md`.

## Ordered, first-match precedence

The policy is a list of rules evaluated top to bottom. The first rule whose
predicate holds selects the action and evaluation stops. The order is
declared as data (`POLICY_ORDER`), so it can be asserted directly rather
than inferred from behaviour.

| # | Policy ID | Action |
|---|---|---|
| 1 | `route.safety.hard_veto` | suppress |
| 2 | `route.risk.pressure_suppress` | suppress |
| 3 | `route.risk.payment_identity_suppress` | suppress |
| 4 | `route.risk.propagation_low_value_suppress` | suppress |
| 5 | `route.urgency.safe_interrupt` | interrupt |
| 6 | `route.value.irrelevant_marketing_suppress` | suppress |
| 7 | `route.value.repetitive_low_value_suppress` | suppress |
| 8 | `route.value.muted_low_value_suppress` | suppress |
| 9 | `route.utility.safe_defer` | defer |
| 10 | `route.analysis.incomplete_defer` | defer |
| — | `route.default.defer` | defer |

**There is no scoring.** No weighted sum, no point total, no probability,
no aggregate. A structural test walks the AST of the whole package and
fails on any arithmetic operator or `sum`/`max`/`min` call, because the
moment features can be added together, a sufficiently "positive" message
can outweigh a risk — which is exactly the failure this design exists to
prevent.

Qualitative levels do have an explicit ordering, used only for local
comparisons inside a single rule. `unknown`, `conflicted`, and `incomplete`
rank with `absent`: none is positive evidence, so none can satisfy a floor
that asks for support.

## Rule 1 — hard veto

When the safety layer reports `hard_veto_required`, the action is
`suppress` and **no later predicate is called at all**. That is what makes
the veto an obligation rather than a strong preference: trust, affinity,
urgency, relevance, usefulness, and classification are not weighed and
lost — they are never reached. A test replaces every later rule with a
tripwire that raises if invoked, and asserts none fires.

Severity and classification are not modified. Contributors record only the
safety finding, so nothing implies that mitigating features were considered
and outweighed.

## Rule 2 — suspicious combinations

Suspicious risk suppresses only in explicit conjunctions, and never by
mutating severity or fabricating a veto:

* **A** suspicious risk **and** high manufactured pressure;
* **B** a payment or credential concern **and** an identity or deceptive-link
  concern;
* **C** high circulation **and** established low relevance **and** established
  low utility, with suspicious risk present.

Each requires two or more independent signals. Alone, none of these
suppresses: a weak finding, a single suspicious indicator, an unverified
domain, missing trust, or forwarding metadata. An unresolved suspicious
case falls through to deferral rather than being forced either way.

## Rule 5 — safe interrupt

Interrupting requires *all* of:

* no hard veto and no suspicious suppression (guaranteed by precedence);
* legitimate urgency `high`;
* recipient relevance at least `medium`;
* malicious pressure not `high`;
* modality consistency not `conflicting`;
* completeness not materially degraded.

None of these alone is enough. Shouting, punctuation, the isolated word
"urgent", direct targeting, high utility, affinity, and trust each fail to
interrupt on their own — verified individually across every feature level.
Malicious pressure can never support an interrupt.

Quiet hours, a silenced conversation, and interruption cost are recorded as
contextual contributors and deliberately do **not** block a genuinely
urgent message. Unknown trust does not block one either.

## Rules 6–8 — low value

Explicit and narrow: irrelevant marketing; heavily circulated content with
**established** low value; a silenced conversation carrying nothing
relevant, useful, or urgent.

"Established" matters. `unknown` means context was never determined and is
not the same finding as `low`. Rule 7 requires at least one positively
established low-value signal, so forwarding metadata alone cannot suppress
a message whose relevance and utility simply were not available.

Alone, none of these suppresses: a promotion, forwarding, a silenced
thread, quiet hours, or absent engagement. A useful promotion defers, and
urgent content in a silenced thread still interrupts.

## Rules 9–11 — deferral

Safe content that is useful or relevant but not urgently so defers
(`route.utility.safe_defer`). Cases that are ambiguous, conflicted, or
built on degraded analysis defer with `route.analysis.incomplete_defer` —
the states that qualify are `incomplete_unavailable_backend`,
`incomplete_unsupported_media`, `incomplete_failed_analysis`, `conflicted`,
`insufficient_content`, and any conflicting or degraded modality
consistency.

Degraded analysis never suppresses and never interrupts on its own:
not knowing is a reason to wait, not a conclusion in either direction. A
reliable severe finding still reaches Rule 1 even when another source
failed.

Anything else defers by default, claiming nothing.

## Classification independence

Classification is one contributor and is never mutated. No class is bound
to an action: the same class reaches `interrupt`, `defer`, and `suppress`
under different safety and feature states, and different classes share an
action. `suspicious_content` alone does not veto.

## Result contract

A result carries the action, one primary policy ID, ordered contributors,
`hard_veto_applied`, `analysis_incomplete`, and fixed diagnostic codes.
Contributors carry a family, a qualitative level, an effect (`supported`,
`prevented`, `contextual`), rule IDs, and a support source.

Deliberately absent: user-facing rationale, confidence, evidence
identifiers or ranking, probability, aggregate score, and any raw content,
URL, credential, payment, health, or path data. `serialize()` returns a
deterministic content-free view for exact comparison.

## Out of scope here

Evidence selection and ranking, final rationale text, final confidence
calibration, any CLI route command, and demonstration data.
