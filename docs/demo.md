# Demo

How to run the router on a request document, read what it decides, and see
six fictional messages take six different paths. Every message in `examples/`
was written for this project; none comes from real traffic or from any
third-party dataset.

## Run it

No installation is needed beyond Python 3.10 or newer; the project has no
dependencies. From the repository root:

```bash
PYTHONPATH=src python -m message_router route --input examples/benign.json
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m message_router route --input examples/benign.json
```

`route` prints one JSON decision to standard output and exits `0`. It adds no
rules of its own: it loads the document, then calls the existing safety
analysis, classification, feature extraction and ordered routing policy (see
[architecture.md](architecture.md) for the exact order).

Installing the package from inside a checkout (`pip install .`, with or
without `-e`) works, but it leaves `build/` and `src/*.egg-info` behind. Both
are git-ignored, yet the public-tree audit (`scripts/audit_public_tree.py`)
reports them as findings until they are deleted, so the `PYTHONPATH` form above
is the one to prefer.

## The request document

A request is one JSON file. The full format is documented in
`src/message_router/io/schema.py`; this is the smallest useful form:

```json
{
  "format": "message-router.request",
  "version": 1,
  "request": {
    "recipient": {"id": "rcp-demo"},
    "conversation": {"id": "conv-demo", "surface": "direct", "participants": 2},
    "sender": {"id": "snd-riverside"},
    "message": {
      "id": "note-benign",
      "sent_at": "2031-03-09T14:30:00+00:00",
      "body": "Lunch on Friday? I booked the usual table at the cafe by the river for twelve thirty."
    }
  }
}
```

The format has no field for an action, category, score or label, so a document
cannot dictate its own outcome. Unknown fields are rejected.

## The decision

```
action                 defer | interrupt | suppress
policy_id              which ordered policy selected the action
classification         what kind of message the classifier judged it to be
hard_veto_applied      true when a severe safety finding short-circuited routing
analysis_incomplete    true when part of the input could not be read
diagnostics            fixed, content-free reasons the policy recorded
contributors           the qualitative feature levels the policy consulted
safety                 the findings: rule id, category, severity, provenance
```

The decision holds identifiers and enumerated values only. It never contains
the message text, and it has no confidence value or free-text explanation.
Both are absent by design, not omitted for now.

## Six demonstrations

| # | Example | Action | Policy | Classification |
|---|---|---|---|---|
| 1 | `examples/benign.json` | `defer` | `route.utility.safe_defer` | `scheduled_event` |
| 2 | `examples/urgent.json` | `interrupt` | `route.urgency.safe_interrupt` | `safety_alert` |
| 3 | `examples/credential_request.json` | `suppress` | `route.safety.hard_veto` | `suspicious_content` |
| 4 | `examples/suspicious_payment.json` | `suppress` | `route.safety.hard_veto` | `suspicious_content` |
| 5 | `examples/ocr_example.json` | `suppress` | `route.safety.hard_veto` | `suspicious_content` |
| 6 | `examples/asr_example.json` | `suppress` | `route.safety.hard_veto` | `suspicious_content` |

Each excerpt below is copied from a real run and trimmed to the fields that
matter; the full output also lists `contributors` and `diagnostics`.

### 1. An ordinary message

*A friend proposes lunch on Friday and says where the table is booked.*

```bash
PYTHONPATH=src python -m message_router route --input examples/benign.json
```

```json
{
  "action": "defer",
  "analysis_incomplete": false,
  "classification": {"ambiguous": false, "selected": "scheduled_event"},
  "hard_veto_applied": false,
  "policy_id": "route.utility.safe_defer",
  "safety": {"findings": [], "highest_severity": null}
}
```

Nothing is wrong with this message and nothing makes it urgent, so it waits:
no finding, relevant to the reader, useful, not time-critical. It shows the
default path, where content that is safe and valuable is neither interrupted
nor discarded.

### 2. A genuinely urgent message

*Building facilities reports a fire alarm and asks everyone to evacuate.*

```bash
PYTHONPATH=src python -m message_router route --input examples/urgent.json
```

```json
{
  "action": "interrupt",
  "analysis_incomplete": false,
  "classification": {"ambiguous": false, "selected": "safety_alert"},
  "hard_veto_applied": false,
  "policy_id": "route.urgency.safe_interrupt",
  "safety": {"findings": [], "highest_severity": null}
}
```

Interrupting requires several independent conditions at once (high legitimate
urgency, enough relevance, no manufactured pressure, no degraded analysis),
and urgent wording alone satisfies none of them. Here the urgency has an
independent origin: the message is classified as a safety alert. It shows that
interrupts are earned, not triggered by words like "now". The rule is in
[routing-policy.md](routing-policy.md).

### 3. A request for a one-time code

*Someone posing as a helpdesk asks the recipient to send the code that was
just received.*

```bash
PYTHONPATH=src python -m message_router route --input examples/credential_request.json
```

```json
{
  "action": "suppress",
  "analysis_incomplete": false,
  "classification": {"ambiguous": true, "selected": "suspicious_content"},
  "hard_veto_applied": true,
  "policy_id": "route.safety.hard_veto",
  "safety": {
    "findings": [
      {
        "category": "credential_solicitation",
        "provenance": "authored_text",
        "rule_id": "CRED_SECRET_DISCLOSURE",
        "severity": "severe"
      }
    ],
    "highest_severity": "severe"
  }
}
```

A severe finding creates a hard-veto obligation. The veto is the first policy,
and when it applies no later rule is evaluated at all, so trust, relevance and
urgency are never weighed against it. The finding names the rule, the category
and where the text came from, and carries none of the text. The classification
being marked ambiguous does not matter here: classification does not select an
action.

### 4. A payment demand pointing at a deceptive link

*A fake courier says a parcel is on hold and asks for a release fee at a raw
address. The address is in a range reserved for documentation, so it points
nowhere real.*

```bash
PYTHONPATH=src python -m message_router route --input examples/suspicious_payment.json
```

```json
{
  "action": "suppress",
  "analysis_incomplete": false,
  "classification": {"ambiguous": true, "selected": "suspicious_content"},
  "hard_veto_applied": true,
  "policy_id": "route.safety.hard_veto",
  "safety": {
    "findings": [
      {
        "category": "coercive_payment",
        "provenance": "authored_text",
        "rule_id": "PAY_URGENT_DEMAND",
        "severity": "severe"
      },
      {
        "category": "deceptive_link",
        "provenance": "authored_text",
        "rule_id": "LNK_RAW_IP",
        "severity": "suspicious"
      }
    ],
    "highest_severity": "severe"
  }
}
```

A payment demand alone is only a concern. It becomes severe because the
demand directs the payment to the deceptive destination, a relationship the
rules establish structurally rather than by two phrases merely appearing in the
same message. The link is judged from its shape and is never fetched.

### 5. Text read from an image (synthetic)

*A recipient gets a screenshot of a "parcel notice". Imagine an OCR engine has
read the text printed on it.*

**No OCR engine is part of this project.** The standard `route` command reads
no image. On this document it reports the attachment as unread:

```bash
PYTHONPATH=src python -m message_router route --input examples/ocr_example.json
```

That gives `defer` with `analysis_incomplete: true` and no findings. To show
what happens to text *after* extraction, a separate runner supplies
**pre-extracted synthetic OCR text** through the project's existing OCR port:

```bash
PYTHONPATH=src python examples/run_media_examples.py --input examples/ocr_example.json
```

```json
{
  "action": "suppress",
  "analysis_incomplete": true,
  "classification": {"ambiguous": true, "selected": "suspicious_content"},
  "hard_veto_applied": true,
  "policy_id": "route.safety.hard_veto",
  "safety": {
    "findings": [
      {
        "category": "coercive_payment",
        "provenance": "ocr_text",
        "rule_id": "PAY_URGENT_DEMAND",
        "severity": "severe"
      },
      {
        "category": "deceptive_link",
        "provenance": "ocr_text",
        "rule_id": "LNK_RAW_IP",
        "severity": "suspicious"
      },
      {
        "category": "malicious_urgency",
        "provenance": "ocr_text",
        "rule_id": "URG_TIME_PRESSURE",
        "severity": "weak"
      }
    ],
    "highest_severity": "severe"
  }
}
```

The same rules run on extracted text as on typed text, and each finding records
its source as `ocr_text`. `analysis_incomplete` is still `true` even though the
text was read: an image has a second reading (scannable codes), no backend was
supplied for it, and the decision says so instead of treating a missing reading
as a clean one.

### 6. Text transcribed from a voice note (synthetic)

*A recipient gets a voicemail. Imagine a speech engine has transcribed it.*

As above, **no speech engine is part of this project** and `route` alone
transcribes nothing. The runner supplies **pre-extracted synthetic ASR text**:

```bash
PYTHONPATH=src python examples/run_media_examples.py --input examples/asr_example.json
```

```json
{
  "action": "suppress",
  "analysis_incomplete": false,
  "classification": {"ambiguous": false, "selected": "suspicious_content"},
  "hard_veto_applied": true,
  "policy_id": "route.safety.hard_veto",
  "safety": {
    "findings": [
      {
        "category": "credential_solicitation",
        "provenance": "asr_transcript",
        "rule_id": "CRED_SECRET_DISCLOSURE",
        "severity": "severe"
      }
    ],
    "highest_severity": "severe"
  }
}
```

The finding's provenance is `asr_transcript`. Here `analysis_incomplete` is
`false` because the only reading that applies to audio, transcription, ran.

### What the media runner is

`examples/run_media_examples.py` contains two fixtures that implement the OCR
and ASR ports. Each returns a fixed, fictional string looked up by attachment
id. No image is opened, no audio is decoded, no recognition happens, and no
image or audio file is included in the repository. The request documents carry
no extracted text, because the format has no field for it; the text comes from
the runner's fixtures, tagged with the provenance a real engine's output would
carry.

## What the demo does not use

* `history` records in a document are validated but are not consulted by
  `route`. Output is identical with or without them.
* `preferences` in a document are used. They feed feature extraction, so a
  silenced conversation or quiet-hours setting changes the corresponding
  contributor levels, and a silenced thread of established low value can be
  suppressed (policy 9). None of the six examples carries preferences, and
  they cannot soften a safety finding.
* The urgency inputs "directly addressed" and "near-term event" exist in the
  analysis stage but have no field in the document format, so `route` cannot
  supply them. `examples/urgent.json` reaches `interrupt` because the message
  itself is classified as a safety alert, which the policy treats as urgent on
  its own.

## Errors

| Situation | Exit status | Output |
|---|---|---|
| Decision produced | `0` | one JSON document on standard output |
| File missing or unreadable | `1` | `error=...` on standard error, nothing on standard output |
| Not valid JSON, or not a valid request document | `1` | same |
| A stage failed while routing | `1` | `error=routing_failed (<ExceptionType>)` |
| Missing `--input` | `2` | usage message on standard error |

Error text names a stage and a location in the document, never a value from it.

## Tests

The commands above are exercised by the test suite, along with checks that
the tables and excerpts in this file match what the router actually prints.
See the testing section of the [README](../README.md) for how to run it.
