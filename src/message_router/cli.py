"""Command-line interface for the message_router portfolio scaffold.

Standard library only. This module currently exposes a placeholder command:
it accepts no subcommands, prints a short status message, and exits
successfully. Future phases will add the actual message-routing behavior.
"""

from __future__ import annotations

import argparse

UNDER_CONSTRUCTION_MESSAGE = (
    "message_router is a portfolio scaffold for a multimodal message-routing "
    "project. The domain foundation (message_router.domain), request-document "
    "loading (message_router.io), recipient-scoped history "
    "(message_router.repositories), and replaceable media adapters "
    "(message_router.media, message_router.ports) are available to import. "
    "The routing stages are still under construction, so this build exposes "
    "only a placeholder command-line interface."
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    print(UNDER_CONSTRUCTION_MESSAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
