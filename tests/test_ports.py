"""Tests for the backend Protocols.

Each test double is an ordinary small class: no base class, no registration,
no patching. That is the point of using structural Protocols -- swapping a
backend never requires importing anything from this package.
"""

from __future__ import annotations

import datetime as dt
import importlib
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from message_router.domain import (  # noqa: E402
    BackendIdentity,
    EvidenceReference,
    ExtractionStatus,
    MediaAnalysis,
    MediaKind,
    MediaReference,
    Provenance,
    RecipientScope,
    TextContent,
)
from message_router.ports import (  # noqa: E402
    AsrBackend,
    Clock,
    EvidenceRepository,
    OcrBackend,
    SemanticAnalysisBackend,
)

READER = RecipientScope("rcp-4kq")
WHEN = dt.datetime(2031, 3, 9, 14, 30, tzinfo=dt.timezone.utc)
EARLIER = dt.datetime(2031, 3, 9, 9, 5, tzinfo=dt.timezone.utc)


# ---------------------------------------------------------------------------
# Test doubles.
# ---------------------------------------------------------------------------


class StubOcr:
    """Returns a fixed reading for any image."""

    def extract_text(self, media: MediaReference) -> MediaAnalysis:
        return MediaAnalysis(
            media,
            ExtractionStatus.SUCCEEDED,
            backend=BackendIdentity("stub-ocr", "0.1"),
            content=TextContent("posted notice about the side gate", Provenance.OCR_TEXT),
        )


class UnavailableOcr:
    """Reports its own absence instead of raising."""

    def extract_text(self, media: MediaReference) -> MediaAnalysis:
        return MediaAnalysis(media, ExtractionStatus.BACKEND_UNAVAILABLE)


class FailingAsr:
    """Reports a controlled failure instead of raising."""

    def transcribe(self, media: MediaReference) -> MediaAnalysis:
        return MediaAnalysis(media, ExtractionStatus.FAILED, backend=BackendIdentity("stub-asr"))


class StubAsr:
    def transcribe(self, media: MediaReference) -> MediaAnalysis:
        return MediaAnalysis(
            media,
            ExtractionStatus.SUCCEEDED,
            backend=BackendIdentity("stub-asr", "0.1"),
            content=TextContent("running late by ten minutes", Provenance.ASR_TRANSCRIPT),
        )


class StubEvidenceRepository:
    def __init__(self, references):
        self._references = tuple(references)
        self.calls = []

    def references_before(self, recipient, moment, limit):
        self.calls.append((recipient, moment, limit))
        return [
            ref
            for ref in self._references
            if ref.belongs_to(recipient) and ref.is_strictly_prior_to(moment)
        ][:limit]


class StubSemanticAnalysis:
    def interpret(self, content: TextContent) -> TextContent:
        return TextContent("a summary produced by a model", Provenance.MODEL_INTERPRETATION)


class FixedClock:
    def __init__(self, moment):
        self._moment = moment

    def now(self):
        return self._moment


# ---------------------------------------------------------------------------
# Protocol conformance and behaviour.
# ---------------------------------------------------------------------------


class ProtocolConformanceTests(unittest.TestCase):
    def test_doubles_satisfy_their_protocols_structurally(self):
        self.assertIsInstance(StubOcr(), OcrBackend)
        self.assertIsInstance(UnavailableOcr(), OcrBackend)
        self.assertIsInstance(StubAsr(), AsrBackend)
        self.assertIsInstance(FailingAsr(), AsrBackend)
        self.assertIsInstance(StubEvidenceRepository(()), EvidenceRepository)
        self.assertIsInstance(StubSemanticAnalysis(), SemanticAnalysisBackend)
        self.assertIsInstance(FixedClock(WHEN), Clock)

    def test_unrelated_object_does_not_satisfy_a_protocol(self):
        class Unrelated:
            pass

        self.assertNotIsInstance(Unrelated(), OcrBackend)

    def test_doubles_subclass_nothing_from_this_package(self):
        for double in (StubOcr, StubAsr, StubEvidenceRepository, StubSemanticAnalysis, FixedClock):
            self.assertEqual(double.__bases__, (object,))


class OcrPortTests(unittest.TestCase):
    def test_successful_reading_carries_ocr_provenance(self):
        analysis = StubOcr().extract_text(MediaReference("att-55c", MediaKind.IMAGE))
        self.assertIs(analysis.status, ExtractionStatus.SUCCEEDED)
        self.assertIs(analysis.content.provenance, Provenance.OCR_TEXT)

    def test_unavailable_backend_is_a_returned_value_not_an_exception(self):
        analysis = UnavailableOcr().extract_text(MediaReference("att-55c", MediaKind.IMAGE))
        self.assertIs(analysis.status, ExtractionStatus.BACKEND_UNAVAILABLE)
        self.assertIsNone(analysis.content)


class AsrPortTests(unittest.TestCase):
    def test_successful_transcript_carries_asr_provenance(self):
        analysis = StubAsr().transcribe(MediaReference("att-77d", MediaKind.AUDIO))
        self.assertIs(analysis.content.provenance, Provenance.ASR_TRANSCRIPT)

    def test_failure_is_a_returned_value_not_an_exception(self):
        analysis = FailingAsr().transcribe(MediaReference("att-77d", MediaKind.AUDIO))
        self.assertIs(analysis.status, ExtractionStatus.FAILED)
        self.assertIsNone(analysis.content)


class EvidenceRepositoryPortTests(unittest.TestCase):
    def test_returns_only_matching_recipient_and_prior_records(self):
        mine = EvidenceReference("rec-3300", READER, EARLIER)
        theirs = EvidenceReference("rec-8801", RecipientScope("rcp-9wz"), EARLIER)
        future = EvidenceReference("rec-9902", READER, WHEN + dt.timedelta(hours=1))
        repo = StubEvidenceRepository((mine, theirs, future))

        returned = repo.references_before(READER, WHEN, 10)

        self.assertEqual([r.record_id for r in returned], ["rec-3300"])
        for reference in returned:
            reference.require_usable_for(READER, WHEN)  # must not raise

    def test_recipient_is_a_required_argument(self):
        repo = StubEvidenceRepository(())
        repo.references_before(READER, WHEN, 5)
        self.assertEqual(repo.calls[0][0], READER)


class SemanticAnalysisPortTests(unittest.TestCase):
    def test_interpretation_is_tagged_as_model_output(self):
        reading = StubSemanticAnalysis().interpret(
            TextContent("original wording", Provenance.AUTHORED_TEXT)
        )
        self.assertIs(reading.provenance, Provenance.MODEL_INTERPRETATION)
        self.assertTrue(reading.provenance.is_extracted)
        self.assertFalse(reading.provenance.carries_instruction_authority)


class ClockPortTests(unittest.TestCase):
    def test_clock_is_deterministic(self):
        clock = FixedClock(WHEN)
        self.assertEqual(clock.now(), clock.now())
        self.assertEqual(clock.now(), WHEN)


# ---------------------------------------------------------------------------
# Dependency independence.
# ---------------------------------------------------------------------------


class DependencyIndependenceTests(unittest.TestCase):
    def test_package_imports_without_optional_media_engines(self):
        """A clean subprocess with the optional engine names blocked must
        still import the whole public surface."""
        script = (
            "import sys\n"
            "blocked = {'cv2', 'PIL', 'numpy', 'pytesseract', 'speech_recognition', 'torch'}\n"
            "class Blocker:\n"
            "    def find_module(self, name, path=None):\n"
            "        if name.split('.')[0] in blocked:\n"
            "            raise ImportError('blocked for test: ' + name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "import message_router\n"
            "import message_router.domain\n"
            "import message_router.ports\n"
            "import message_router.validation\n"
            "print('IMPORT_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
            env={**__import__("os").environ, "PYTHONPATH": str(SRC_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("IMPORT_OK", result.stdout)

    def test_no_third_party_module_is_imported_by_the_package(self):
        stdlib = set(sys.stdlib_module_names)
        for module_name in (
            "message_router.domain",
            "message_router.ports",
            "message_router.validation",
        ):
            importlib.import_module(module_name)
        for name, module in list(sys.modules.items()):
            if not name.startswith("message_router"):
                continue
            for attr in vars(module).values():
                origin = getattr(attr, "__module__", "")
                root = str(origin).split(".")[0]
                if root and root not in stdlib and not root.startswith("message_router"):
                    self.fail(f"{name} exposes a non-stdlib dependency: {root}")


if __name__ == "__main__":
    unittest.main()
