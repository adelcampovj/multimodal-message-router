"""Command-line interface for the message_router portfolio scaffold.

Standard library only. Two behaviours exist today:

* no subcommand -- print a short status message and exit successfully;
* ``validate`` -- load a request document and report whether it is
  structurally valid.

``validate`` is a *loading* check only. It deliberately does not run safety
analysis and does not route anything: it answers "would this document parse
into valid objects", nothing more. Routing has not been implemented yet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

UNDER_CONSTRUCTION_MESSAGE = (
    "message_router is a portfolio scaffold for a multimodal message-routing "
    "project. The domain foundation (message_router.domain), request-document "
    "loading (message_router.io), recipient-scoped history "
    "(message_router.repositories), replaceable media adapters "
    "(message_router.media, message_router.ports), and deterministic safety "
    "analysis (message_router.safety) are available to import. The routing "
    "stages are still under construction, so this build exposes only a "
    "placeholder command-line interface."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="message_router",
        description=(
            "Multimodal message-routing portfolio project. Intended to "
            "classify incoming messages for immediate, later, or suppressed "
            "delivery based on message content and context."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    validate = subparsers.add_parser(
        "validate",
        help="check that a request document loads into valid objects",
        description=(
            "Load a request document and report whether it is structurally "
            "valid. Performs no safety analysis and no routing."
        ),
    )
    validate.add_argument("scenario", help="path to a request document")
    return parser


def _validate(scenario: str) -> int:
    from .io import LoadError, load_request_file

    try:
        loaded = load_request_file(Path(scenario))
    except LoadError as exc:
        # The load error is already sanitized: a stage, a reason, and a
        # pointer into the document -- never a value from it.
        print("result=INVALID")
        print(f"reason={exc}")
        return 1

    print("result=VALID")
    print(f"has_authored_text={int(loaded.message.has_authored_text)}")
    print(f"has_attachment={int(loaded.message.has_attachment)}")
    print(f"history_records={len(loaded.history)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        return _validate(args.scenario)
    print(UNDER_CONSTRUCTION_MESSAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
