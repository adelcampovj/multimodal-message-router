# multimodal-message-router

A deterministic, policy-based router that decides whether an incoming message
should **interrupt** the reader now, **defer** until later, or be
**suppressed**, treating text, image-derived text and voice-derived text as
untrusted input.

It is written in standard-library Python with no dependencies, runs from a
single command, and every decision carries an inspectable trace: a policy
identifier and a set of content-free findings.

This is an independent portfolio implementation inspired by a hackathon
problem about WhatsApp-style message notifications. It is not an official
submission, has no connection to the event's organisers, and contains no data
from it. Every example message here is fictional.

## The problem

A notification system that treats all messages alike either buries the urgent
one or interrupts for noise. Worse, messages are written by other people, and
some of those people want to be treated as urgent: a fake helpdesk asking for a
one-time code, a "courier" demanding a fee at a strange link. Routing is
therefore a safety problem as well as a ranking problem.

This project makes the safety decision explicit. Message content, including
text read out of an image or a voice note, is data to be analysed and never an
instruction to be followed. The route is chosen by a short, ordered, readable
policy rather than a learned score, so any decision can be traced to the rule
that made it.

## What it does

| Capability | What it means here |
|---|---|
| Strict input | One versioned JSON request document. Unknown fields are rejected, and the format has no field for an action or score, so a document cannot dictate its own outcome. |
| Safety analysis | Pattern-based findings across 11 categories (credential requests, coercive payment, deceptive links, prompt injection and others) at three severities. Findings carry a rule id, category, severity and provenance, never the matched text. |
| Classification | 10 message classes, with ambiguity kept rather than forced. |
| Features | 15 feature families with qualitative levels, where `unknown` is deliberately different from `absent`. |
| Ordered routing | 13 ordered policies plus a default. The first match selects one of `interrupt`, `defer` or `suppress`. |
| Hard veto | A severe finding forces `suppress`, and no later rule is consulted. |
| Provenance | Authored text, OCR text, speech transcripts and decoded codes are tracked separately and never gain instruction authority. |
| Replaceable extraction | OCR, ASR and QR are interfaces. No recognition engine is included. |
| CLI | `validate` checks a document loads. `route` prints one JSON decision. |

## How it works

```mermaid
flowchart LR
    A(["Request JSON"]) --> B["Validate<br/>strict schema"]
    B --> C["Safety analysis<br/>findings and hard-veto flag"]
    C --> D["Classify<br/>10 classes"]
    D --> E["Features<br/>15 qualitative families"]
    E --> F["Ordered policy<br/>13 rules and a default"]
    F --> G(["interrupt, defer or suppress"])
    M["Attachment readings<br/>OCR, ASR, QR backends<br/>none included"] -.-> C
```

A request is validated into typed objects, any attachment readings are
collected from whatever backends the caller supplies, and the pipeline runs
safety analysis, then classification, then feature extraction, then the policy.
The `route` command supplies no backends, so it reads no images and transcribes
no audio. The full diagram, with trust boundaries and the three ways media can
be handled, is in [docs/architecture.md](docs/architecture.md). The policy
itself is in [docs/routing-policy.md](docs/routing-policy.md).

## Engineering design highlights

* **A veto, not a weight.** The hard veto is the first policy. When it applies,
  no later predicate runs, which a test enforces by replacing every later rule
  with one that raises. Trust, relevance or urgency are never weighed against a
  severe finding.
* **No scores anywhere.** A structural test walks the routing package and fails
  on arithmetic or `sum`, `max` and `min`, so a sufficiently "positive" feature
  cannot outweigh a risk.
* **Corroboration counts causes, not repetition.** Two findings that express
  the same underlying indicator count once, so the same demand read from typed
  text and from OCR text cannot corroborate itself.
* **Untrusted by construction.** Every provenance is recorded and none can
  carry instruction authority or confer verified trust. Tests iterate the whole
  enumeration to prove it.
* **Honest about what it could not read.** A failed or unavailable reading
  marks the analysis incomplete. It is never treated as evidence of safety.
* **Deterministic and bounded.** No randomness, clock, network or model call.
  Analysis is bounded at 20,000 characters per source, and tests confirm that
  inputs with tens of thousands of repeated separators complete promptly.
* **Claims are tested.** Documentation tests check that links resolve, diagrams
  show the stages in the order the code calls them, and the tables and JSON
  excerpts here match what the router prints.
* **A public-tree audit.** `scripts/audit_public_tree.py` keeps data files,
  media, secrets, local paths and build artefacts out of the repository, and
  runs in CI.

## Quickstart

Requires Python 3.10 or newer. It was developed and tested with Python 3.12 and
needs no packages. Run from the repository root; nothing needs installing.

```bash
PYTHONPATH=src python -m message_router --help
PYTHONPATH=src python -m message_router route --help
PYTHONPATH=src python -m message_router validate examples/benign.json
PYTHONPATH=src python -m message_router route --input examples/credential_request.json
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m message_router --help
python -m message_router route --input examples/credential_request.json
```

The last command prints one JSON decision and exits `0`. Invalid input exits
non-zero with an error that names a stage and a location in the document, never
a value from it.

If you want an installed copy, install from a copy of `pyproject.toml` and
`src/` made outside the checkout, which is what CI does. Installing from inside
the checkout, editable or not, leaves `build/` and `src/*.egg-info` behind.
Both are git-ignored, but the public-tree audit fails until they are deleted.

## Demonstrations

Six fictional messages, each a runnable command. Details, scenarios and JSON
excerpts are in [docs/demo.md](docs/demo.md).

| Example | Action | Policy | Classification |
|---|---|---|---|
| `examples/benign.json` | `defer` | `route.utility.safe_defer` | `scheduled_event` |
| `examples/urgent.json` | `interrupt` | `route.urgency.safe_interrupt` | `safety_alert` |
| `examples/credential_request.json` | `suppress` | `route.safety.hard_veto` | `suspicious_content` |
| `examples/suspicious_payment.json` | `suppress` | `route.safety.hard_veto` | `suspicious_content` |
| `examples/ocr_example.json` | `suppress` | `route.safety.hard_veto` | `suspicious_content` |
| `examples/asr_example.json` | `suppress` | `route.safety.hard_veto` | `suspicious_content` |

The first four use `route` directly:

```bash
PYTHONPATH=src python -m message_router route --input examples/benign.json
PYTHONPATH=src python -m message_router route --input examples/urgent.json
PYTHONPATH=src python -m message_router route --input examples/credential_request.json
PYTHONPATH=src python -m message_router route --input examples/suspicious_payment.json
```

The last two need the example runner, because the request format has no field
for extracted text:

```bash
PYTHONPATH=src python examples/run_media_examples.py --input examples/ocr_example.json
PYTHONPATH=src python examples/run_media_examples.py --input examples/asr_example.json
```

These two use **pre-extracted synthetic OCR and ASR text**: fixed fictional
strings supplied through the project's OCR and ASR interfaces, tagged with the
provenance a real engine's output would carry. No image is opened and no audio
is decoded. There is no OCR or speech engine in this repository. Run on its own,
`route` reports such an attachment as unread, defers, and sets
`analysis_incomplete` to `true`.

## Testing and CI

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python scripts/audit_public_tree.py
```

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest discover -s tests -v
python scripts/audit_public_tree.py
```

The environment variable matters: some tests start subprocesses, and without it
Python writes `__pycache__` directories that the audit rejects. (`python -B`
alone is not enough, because it does not reach those subprocesses.)

At this commit the suite has 1,005 tests: 999 pass and 6
skip, with no failures or errors, and the audit reports no findings. The skips
are environmental: on the Windows machine used for development, six tests are
skipped because creating symbolic links needs a privilege that was not
available.

These are **synthetic** tests. Inputs are fictional and the expected results
encode intended behaviour, covering domain contracts, the loader, each safety
rule family, the routing policy, structural guards (no scoring, no network
imports), the CLI and examples, the audit itself, and these documents. They
show that designed behaviour holds and that regressions are caught. They are
not an accuracy measurement: there is no labelled real-world corpus here, so
this project reports no accuracy, precision or recall.

A GitHub Actions workflow in `.github/workflows/ci.yml` runs the tests, the
audit and a CLI check on Ubuntu and Windows with Python 3.12. This repository
has not been published, so that workflow has not run on a hosted runner.

## Known limitations

* **A deterministic, policy-based prototype.** Detection is finite and
  pattern-based, tuned to English. It will miss paraphrases, other languages
  and novel phrasings, and a message with no findings has not been shown to be
  safe. It is not production-certified.
* **No recognition is included.** The media examples use pre-extracted
  synthetic text. Real OCR, ASR and QR integration needs external backends
  written against the existing interfaces, and the `route` command supplies none.
* **`history` is validated but unused.** The request format accepts history
  records, but they do not influence the decision.
* **`preferences` are used, narrowly.** They reach feature extraction, so a
  silenced conversation or quiet-hours setting changes the matching
  contributors, and a silenced low-value thread can be suppressed. They cannot
  soften a safety finding.
* **Some benign requests are suppressed.** A request for a document about
  credentials (for example a "password policy") is treated like a request for
  the credential itself. This fails toward caution and is not yet corrected.
* **Some anti-fraud warnings are suppressed.** "Never share your passwords with
  anyone" is read as a request, and so are the same sentences with "passcodes"
  and "OTPs", because the warning recognition is written around singular
  objects.
* **Long messages are only partly read.** Content past 20,000 characters is not
  analysed. The message is deferred with `analysis_incomplete` set, but a
  hostile request placed after that point is not recognised.
* **No numeric confidence.** The decision exposes no calibrated confidence
  value.
* **No natural-language reason.** The decision exposes a policy id, findings
  and diagnostic codes, but no generated explanation.
* **Synthetic coverage is not real-world accuracy.** Passing tests do not say
  how well the rules would perform on real traffic.

## Future improvements

* Real OCR, ASR and QR adapters behind the existing interfaces, with the
  extracted text carried in the request or supplied by a runner.
* Letting a request supply "directly addressed" and "near-term event" context,
  and using `history` in routing, with tests that show the effect.
* A bounded way to tell a request for a credential from a request for a
  document about one, and warning recognition that covers plural objects.
* Human-readable reasons derived from the policy id and findings.
* Confidence values, only once there is labelled data to calibrate them against.

## Documentation

* [docs/architecture.md](docs/architecture.md): the pipeline, trust boundaries
  and layers, with diagrams.
* [docs/routing-policy.md](docs/routing-policy.md): the ordered policy, rule by
  rule.
* [docs/threat-model.md](docs/threat-model.md): the safety layer, what it
  detects, and what it deliberately does not.
* [docs/demo.md](docs/demo.md): the six demonstrations in detail.
