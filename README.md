# multimodal-message-router

A multimodal message-routing portfolio project.

The intent is a system that looks at an incoming message — across text,
image, and voice content — and decides how it should be handled: delivered
immediately, held for later review, or suppressed entirely.

**Status: this repository is currently a scaffold.** No routing logic has
been implemented yet; the package exposes only a placeholder command-line
interface while the rest of the project is built out.

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
