"""Command-line interface for the message_router portfolio project.

Standard library only. Three behaviours exist today:

* no subcommand -- print a short status message and exit successfully;
* ``validate`` -- load a request document and report whether it is
  structurally valid;
* ``route`` -- load a request document, run it through the existing
  analysis and routing stages, and print one JSON decision.

``validate`` is a *loading* check only. It deliberately does not run safety
analysis and does not route anything: it answers "would this document parse
into valid objects", nothing more.

``route`` adds no rule of its own; it calls
:func:`message_router.pipeline.route_request`. It performs no image
recognition and no speech transcription: an attachment in the document is
reported as unread, and the decision says its analysis is incomplete.
"""

from __future__ import annotations

import argparse
import json
import sys
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

    route = subparsers.add_parser(
        "route",
        help="route one request document and print a JSON decision",
        description=(
            "Load a request document, run it through the existing safety, "
            "classification and routing stages, and print one JSON decision "
            "to standard output. The decision contains identifiers and "
            "enumerated values only, never the message text. Attachments are "
            "not read: no image recognition or transcription is performed."
        ),
    )
    route.add_argument(
        "--input", required=True, metavar="PATH", help="path to a request document"
    )
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


def _route(path: str) -> int:
    from .io import LoadError, load_request_file
    from .pipeline import decision_to_dict, route_request

    try:
        loaded = load_request_file(Path(path))
    except LoadError as exc:
        # Sanitized at the source: a stage, a reason and a pointer into the
        # document, never a value from it. Errors go to stderr so standard
        # output carries a decision or nothing.
        print(f"error={exc}", file=sys.stderr)
        return 1

    try:
        decision = decision_to_dict(route_request(loaded))
    except Exception as exc:
        # The exception type is reported; its text is not, because it could
        # carry a fragment of the request.
        print(f"error=routing_failed ({type(exc).__name__})", file=sys.stderr)
        return 1

    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        return _validate(args.scenario)
    if args.command == "route":
        return _route(args.input)
    print(UNDER_CONSTRUCTION_MESSAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
