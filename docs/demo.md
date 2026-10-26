# Demo

How to run the router on a request document and read what it decides. Every
message in `examples/` is fictional and was written for this project.

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

If you installed the package with `pip install -e .`, drop the `PYTHONPATH`.

`route` prints one JSON decision to standard output and exits `0`. It adds no
rules of its own: it loads the document, then calls the existing safety
analysis, classification, feature extraction and ordered routing policy.

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

## The six examples

Actual output of `route` (and, for the last two, the media runner below):

| Example | Action | Policy | Classification |
|---|---|---|---|
| `examples/benign.json` | `defer` | `route.utility.safe_defer` | `scheduled_event` |
| `examples/urgent.json` | `interrupt` | `route.urgency.safe_interrupt` | `safety_alert` |
| `examples/credential_request.json` | `suppress` | `route.safety.hard_veto` | `suspicious_content` |
| `examples/suspicious_payment.json` | `suppress` | `route.safety.hard_veto` | `suspicious_content` |
| `examples/ocr_example.json` | `suppress` | `route.safety.hard_veto` | `suspicious_content` |
| `examples/asr_example.json` | `suppress` | `route.safety.hard_veto` | `suspicious_content` |

The payment example uses `203.0.113.9`, an address reserved for documentation
(RFC 5737), so it points nowhere real.

## Images and voice: what is and is not real

**The `route` command reads no images and transcribes no audio.** Run it on
`examples/ocr_example.json` or `examples/asr_example.json` and the attachment
is reported as unread:

```bash
PYTHONPATH=src python -m message_router route --input examples/ocr_example.json
```

The result is `defer` with `analysis_incomplete: true` and no findings.

To show how text that came *from* an image or a recording is routed, the
runner `examples/run_media_examples.py` supplies it through the project's
existing `OcrBackend` and `AsrBackend` ports:

```bash
PYTHONPATH=src python examples/run_media_examples.py --input examples/ocr_example.json
PYTHONPATH=src python examples/run_media_examples.py --input examples/asr_example.json
```

The two backends in that script are fixtures. Each returns a fixed, fictional
string, **pre-extracted synthetic OCR text** and **pre-extracted synthetic ASR
text**, looked up by attachment id. No image is opened, no audio is decoded,
no recognition happens, and no image or audio file is included in the
repository. The strings carry the provenance real engine output would carry,
`ocr_text` and `asr_transcript`, and the findings in the decision show it:

* `ocr_example` is routed from `ocr_text`: `PAY_URGENT_DEMAND`,
  `LNK_RAW_IP` and `URG_TIME_PRESSURE`.
* `asr_example` is routed from `asr_transcript`: `CRED_SECRET_DISCLOSURE`.

The OCR example also reports `analysis_incomplete: true`. The text was read,
but an image has a second reading (scannable codes) and no backend was supplied
for it, so the decision says so.

## What the demo does not use

* `history` records in a document are validated but are not consulted by
  `route`.
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

```bash
python -m unittest discover -s tests -v
python scripts/audit_public_tree.py
```
