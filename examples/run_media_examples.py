"""Route a media example using PRE-EXTRACTED SYNTHETIC OCR / ASR text.

What this is
------------
The ``route`` command reads no images and transcribes no audio, and the
request-document format has no field for extracted text. To show how text
that came *from* an image or a recording is routed, this script supplies it
the way a real engine would: through the project's existing ``OcrBackend`` and
``AsrBackend`` ports, driven by ``MediaCoordinator``.

What this is not
----------------
The two backends below are fixtures. Each returns a fixed, fictional string
written for this demonstration, looked up by attachment id. No image is
opened, no audio is decoded, and no recognition happens. The strings are
tagged with the provenance a real engine's output would carry
(``ocr_text`` / ``asr_transcript``), which is the point: routing treats
text read out of media as untrusted content from its own source.

Usage::

    python examples/run_media_examples.py --input examples/ocr_example.json
    python examples/run_media_examples.py --input examples/asr_example.json

Prints one JSON decision, in the same shape as ``message_router route``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from message_router.domain import (  # noqa: E402
    ExtractionStatus,
    MediaAnalysis,
    MediaReference,
    Provenance,
    TextContent,
)
from message_router.domain.media import BackendIdentity  # noqa: E402
from message_router.io import LoadError, load_request_file  # noqa: E402
from message_router.media import MediaCoordinator  # noqa: E402
from message_router.pipeline import decision_to_dict, route_request  # noqa: E402

#: Fictional text, authored for this demo, keyed by attachment id.
SYNTHETIC_OCR_TEXT = {
    "att-synthetic-image": (
        "PARCEL NOTICE. A release fee is overdue. "
        "Pay now at http://203.0.113.9/pay or your parcel will be returned."
    ),
}
SYNTHETIC_ASR_TEXT = {
    "att-synthetic-voice": (
        "Hello, this is the helpdesk. Please read out the one time code "
        "we just texted you so we can verify your account."
    ),
}


class SyntheticOcrFixture:
    """Implements ``OcrBackend``. Returns fixed text; reads no image."""

    _IDENTITY = BackendIdentity("synthetic-ocr-fixture", "1")

    def extract_text(self, media: MediaReference) -> MediaAnalysis:
        text = SYNTHETIC_OCR_TEXT.get(media.media_id)
        if text is None:
            return MediaAnalysis(media, ExtractionStatus.FAILED, backend=self._IDENTITY)
        return MediaAnalysis(
            media,
            ExtractionStatus.SUCCEEDED,
            backend=self._IDENTITY,
            content=TextContent(text, Provenance.OCR_TEXT),
        )


class SyntheticAsrFixture:
    """Implements ``AsrBackend``. Returns fixed text; decodes no audio."""

    _IDENTITY = BackendIdentity("synthetic-asr-fixture", "1")

    def transcribe(self, media: MediaReference) -> MediaAnalysis:
        text = SYNTHETIC_ASR_TEXT.get(media.media_id)
        if text is None:
            return MediaAnalysis(media, ExtractionStatus.FAILED, backend=self._IDENTITY)
        return MediaAnalysis(
            media,
            ExtractionStatus.SUCCEEDED,
            backend=self._IDENTITY,
            content=TextContent(text, Provenance.ASR_TRANSCRIPT),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Route a media example using pre-extracted SYNTHETIC OCR/ASR text."
    )
    parser.add_argument("--input", required=True, metavar="PATH", help="path to a request document")
    args = parser.parse_args(argv)

    try:
        loaded = load_request_file(Path(args.input))
    except LoadError as exc:
        print(f"error={exc}", file=sys.stderr)
        return 1

    attachment = loaded.message.attachment
    if attachment is None or attachment.media_id not in (*SYNTHETIC_OCR_TEXT, *SYNTHETIC_ASR_TEXT):
        print("error=no_synthetic_text_for_this_document", file=sys.stderr)
        return 1

    coordinator = MediaCoordinator(ocr=SyntheticOcrFixture(), asr=SyntheticAsrFixture())
    try:
        decision = decision_to_dict(route_request(loaded, coordinator=coordinator))
    except Exception as exc:
        print(f"error=routing_failed ({type(exc).__name__})", file=sys.stderr)
        return 1

    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
