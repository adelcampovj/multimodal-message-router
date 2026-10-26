# multimodal-message-router

A multimodal message-routing portfolio project.

The intent is a system that looks at an incoming message — across text,
image, and voice content — and decides how it should be handled: delivered
immediately, held for later review, or suppressed entirely.

**Status: the routing pipeline runs end to end.** A request document can be
routed from the command line with `route`, which prints one JSON decision.
Image and voice handling is demonstrated with pre-extracted synthetic text
only; no recognition engine is included. See [docs/demo.md](docs/demo.md).
Other documents in this repository are still being brought up to date.

## Development setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .
```

## Running the CLI

```bash
python -m message_router --help
```

## Trying the router

Route one of the six fictional examples (see [docs/demo.md](docs/demo.md)):

```bash
PYTHONPATH=src python -m message_router route --input examples/credential_request.json
```

## Running tests

```bash
python -m unittest discover -s tests -v
```

## Public-tree audit

Before anything is published from this repository, run the audit that
checks the tree for content that should never leave a private working copy:

```bash
python scripts/audit_public_tree.py
```
