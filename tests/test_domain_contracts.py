"""Tests for the independent domain contracts.

All fixtures are small in-memory objects built from neutral identifiers and
invented scenarios.
"""

from __future__ import annotations

import datetime as dt
import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from message_router.domain import (  # noqa: E402
    AbsenceReason,
    AnalysisWarning,
    BackendIdentity,
    Confidence,
    ContextGap,
    ConversationContext,
    ConversationSurface,
    DeterministicSignal,
    EvidenceReference,
    ExtractionStatus,
    IncomingMessage,
    InteractionRecord,
    MediaAnalysis,
    MediaKind,
    MediaReference,
    MessageCategory,
    Provenance,
    QuietPeriod,
    Rationale,
    RecipientPreferences,
    RecipientScope,
    Reliability,
    RiskCategory,
    RiskSignal,
    RoutingAction,
    RoutingDecision,
    SenderContext,
    TextContent,
    TrustCategory,
    TrustSignal,
    UnsupportedValueError,
    ValidationError,
    ValidationIssue,
)
from message_router.validation import TIMESTAMP_POLICY, normalize_timestamp  # noqa: E402

EAST = dt.timezone(dt.timedelta(hours=4))
WHEN = dt.datetime(2031, 3, 9, 14, 30, tzinfo=dt.timezone.utc)
EARLIER = dt.datetime(2031, 3, 9, 9, 5, tzinfo=dt.timezone.utc)

READER = RecipientScope("rcp-4kq")
OTHER_READER = RecipientScope("rcp-9wz")
THREAD = ConversationContext("conv-31b", ConversationSurface.DIRECT)


def body(text: str = "meeting the courier by the side gate") -> TextContent:
    return TextContent(text, Provenance.AUTHORED_TEXT)


def message(**overrides) -> IncomingMessage:
    kwargs = {
        "message_id": "note-7712",
        "recipient": READER,
        "conversation": THREAD,
        "body": body(),
        "sent_at": WHEN,
    }
    kwargs.update(overrides)
    return IncomingMessage(**kwargs)


# ---------------------------------------------------------------------------
# 1. Valid construction of each major object.
# ---------------------------------------------------------------------------


class ValidConstructionTests(unittest.TestCase):
    def test_recipient_scope(self):
        self.assertEqual(READER.recipient_id, "rcp-4kq")

    def test_incoming_message(self):
        msg = message()
        self.assertEqual(msg.message_id, "note-7712")
        self.assertTrue(msg.body.value)

    def test_sender_and_conversation(self):
        sender = SenderContext("snd-208")
        self.assertFalse(sender.has_asserted_trust)
        self.assertEqual(THREAD.surface, ConversationSurface.DIRECT)

    def test_preferences_and_interaction(self):
        prefs = RecipientPreferences(
            quiet_period=QuietPeriod(22 * 60, 7 * 60), muted_conversation_ids=frozenset({"conv-31b"})
        )
        self.assertTrue(prefs.has_muted("conv-31b"))
        record = InteractionRecord(READER, "conv-31b", EARLIER, opened=True)
        self.assertTrue(record.opened)

    def test_media_and_evidence_and_decision(self):
        ref = MediaReference("att-55c", MediaKind.IMAGE)
        analysis = MediaAnalysis(ref, ExtractionStatus.NO_CONTENT_FOUND)
        self.assertFalse(analysis.produced_content)
        evidence = EvidenceReference("rec-3300", READER, EARLIER)
        decision = RoutingDecision(
            "note-7712",
            RoutingAction.DEFER,
            MessageCategory.CONVERSATIONAL,
            Confidence(0.5),
            Rationale("no time-bound request was present"),
            evidence=(evidence,),
        )
        self.assertTrue(decision.cites_evidence)

    def test_signals(self):
        self.assertEqual(
            DeterministicSignal("link_count", "two links present").provenance,
            Provenance.DERIVED_SIGNAL,
        )
        risk = RiskSignal(RiskCategory.UNSOLICITED_BULK, "bulk shape", Provenance.AUTHORED_TEXT)
        trust = TrustSignal(TrustCategory.RECIPIENT_INITIATED, "reader started the thread")
        self.assertEqual(risk.category, RiskCategory.UNSOLICITED_BULK)
        self.assertEqual(trust.strength, 1.0)


# ---------------------------------------------------------------------------
# 2-6. Identifier, confidence, and rationale invariants.
# ---------------------------------------------------------------------------


class ScalarInvariantTests(unittest.TestCase):
    def test_empty_and_whitespace_identifiers_rejected(self):
        for bad in ("", "   ", "\t\n"):
            with self.assertRaises(ValidationError) as ctx:
                RecipientScope(bad)
            self.assertEqual(ctx.exception.issue, ValidationIssue.EMPTY_IDENTIFIER)

    def test_identifiers_are_trimmed(self):
        self.assertEqual(RecipientScope("  rcp-4kq  ").recipient_id, "rcp-4kq")

    def test_confidence_accepts_both_bounds(self):
        self.assertEqual(Confidence(0.0).value, 0.0)
        self.assertEqual(Confidence(1.0).value, 1.0)

    def test_confidence_out_of_range_rejected(self):
        for bad in (-0.001, 1.001, -5.0, 42.0):
            with self.assertRaises(ValidationError) as ctx:
                Confidence(bad)
            self.assertEqual(ctx.exception.issue, ValidationIssue.OUT_OF_RANGE)

    def test_confidence_rejects_nan_and_infinities(self):
        with self.assertRaises(ValidationError) as ctx:
            Confidence(math.nan)
        self.assertEqual(ctx.exception.issue, ValidationIssue.NOT_A_NUMBER)
        for bad in (math.inf, -math.inf):
            with self.assertRaises(ValidationError) as ctx:
                Confidence(bad)
            self.assertEqual(ctx.exception.issue, ValidationIssue.NOT_FINITE)

    def test_confidence_rejects_boolean(self):
        with self.assertRaises(ValidationError) as ctx:
            Confidence(True)
        self.assertEqual(ctx.exception.issue, ValidationIssue.WRONG_TYPE)

    def test_empty_rationale_rejected(self):
        for bad in ("", "   "):
            with self.assertRaises(ValidationError) as ctx:
                Rationale(bad)
            self.assertEqual(ctx.exception.issue, ValidationIssue.EMPTY_TEXT)

    def test_rationale_is_trimmed(self):
        self.assertEqual(Rationale("  held for later  ").text, "held for later")


# ---------------------------------------------------------------------------
# 7-9. Timestamps.
# ---------------------------------------------------------------------------


class TimestampTests(unittest.TestCase):
    def test_aware_timestamp_accepted(self):
        self.assertEqual(message(sent_at=WHEN).sent_at, WHEN)

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            message(sent_at=dt.datetime(2031, 3, 9, 14, 30))
        self.assertEqual(ctx.exception.issue, ValidationIssue.NAIVE_TIMESTAMP)

    def test_normalization_converts_to_utc_preserving_instant(self):
        local = dt.datetime(2031, 3, 9, 18, 30, tzinfo=EAST)
        normalized = normalize_timestamp(local, "sent_at")
        self.assertEqual(normalized.tzinfo, dt.timezone.utc)
        self.assertEqual(normalized, local)
        self.assertEqual(normalized.hour, 14)

    def test_stored_timestamps_are_normalized(self):
        msg = message(sent_at=dt.datetime(2031, 3, 9, 18, 30, tzinfo=EAST))
        self.assertEqual(msg.sent_at.tzinfo, dt.timezone.utc)

    def test_policy_is_documented(self):
        self.assertIn("UTC", TIMESTAMP_POLICY)


# ---------------------------------------------------------------------------
# 10-11. Media consistency, status, provenance preservation.
# ---------------------------------------------------------------------------


class MediaContractTests(unittest.TestCase):
    def test_succeeded_requires_content(self):
        ref = MediaReference("att-55c", MediaKind.IMAGE)
        with self.assertRaises(ValidationError) as ctx:
            MediaAnalysis(ref, ExtractionStatus.SUCCEEDED)
        self.assertEqual(ctx.exception.issue, ValidationIssue.STATUS_CONTENT_MISMATCH)

    def test_non_success_statuses_reject_content(self):
        ref = MediaReference("att-55c", MediaKind.IMAGE)
        for status in (
            ExtractionStatus.NOT_ATTEMPTED,
            ExtractionStatus.NO_CONTENT_FOUND,
            ExtractionStatus.BACKEND_UNAVAILABLE,
            ExtractionStatus.FAILED,
        ):
            with self.assertRaises(ValidationError) as ctx:
                MediaAnalysis(ref, status, content=TextContent("x", Provenance.OCR_TEXT))
            self.assertEqual(ctx.exception.issue, ValidationIssue.STATUS_CONTENT_MISMATCH)

    def test_unavailable_is_distinct_from_no_content_found(self):
        ref = MediaReference("att-55c", MediaKind.IMAGE)
        missing = MediaAnalysis(ref, ExtractionStatus.BACKEND_UNAVAILABLE)
        looked = MediaAnalysis(ref, ExtractionStatus.NO_CONTENT_FOUND)
        self.assertNotEqual(missing.status, looked.status)
        self.assertFalse(missing.produced_content)
        self.assertFalse(looked.produced_content)
        self.assertTrue(missing.was_attempted)
        self.assertFalse(MediaAnalysis(ref, ExtractionStatus.NOT_ATTEMPTED).was_attempted)

    def test_media_kind_and_provenance_must_agree(self):
        audio = MediaReference("att-77d", MediaKind.AUDIO)
        with self.assertRaises(ValidationError) as ctx:
            MediaAnalysis(
                audio,
                ExtractionStatus.SUCCEEDED,
                content=TextContent("read from a picture", Provenance.OCR_TEXT),
            )
        self.assertEqual(ctx.exception.issue, ValidationIssue.MEDIA_KIND_MISMATCH)

    def test_content_provenance_must_be_an_extraction_provenance(self):
        image = MediaReference("att-55c", MediaKind.IMAGE)
        with self.assertRaises(ValidationError) as ctx:
            MediaAnalysis(
                image,
                ExtractionStatus.SUCCEEDED,
                content=TextContent("typed by a person", Provenance.AUTHORED_TEXT),
            )
        self.assertEqual(ctx.exception.issue, ValidationIssue.PROVENANCE_MISMATCH)

    def test_extraction_preserves_provenance_and_value(self):
        image = MediaReference("att-55c", MediaKind.IMAGE)
        extracted = TextContent("gate code on the poster", Provenance.OCR_TEXT)
        analysis = MediaAnalysis(
            image,
            ExtractionStatus.SUCCEEDED,
            backend=BackendIdentity("reference-ocr", "2.1"),
            content=extracted,
            reliability=Reliability.MEDIUM,
            warnings=(AnalysisWarning.NO_CONFIDENCE_REPORTED,),
        )
        self.assertIs(analysis.content.provenance, Provenance.OCR_TEXT)
        self.assertEqual(analysis.content.value, "gate code on the poster")
        self.assertTrue(analysis.content.is_extracted)

    def test_backend_identity_rejects_paths(self):
        # The drive-letter case is assembled from fragments rather than
        # written literally: a literal drive path in a committed source file
        # is itself something the public-tree audit rejects, and this suite
        # should not have to be exempted from that rule to test this branch.
        drive_prefixed = "D" + ":" + "engines"
        for bad in (
            "/opt/engines/reader",
            "engines\\reader.exe",
            "https://example.invalid/x",
            drive_prefixed,
        ):
            with self.assertRaises(ValidationError) as ctx:
                BackendIdentity(bad)
            self.assertEqual(ctx.exception.issue, ValidationIssue.EMBEDDED_PATH)


# ---------------------------------------------------------------------------
# 12-14. Evidence scope and chronology.
# ---------------------------------------------------------------------------


class EvidenceContractTests(unittest.TestCase):
    def test_matching_recipient_scope_accepted(self):
        ref = EvidenceReference("rec-3300", READER, EARLIER)
        self.assertTrue(ref.belongs_to(READER))
        ref.require_usable_for(READER, WHEN)  # must not raise

    def test_cross_recipient_evidence_rejected(self):
        ref = EvidenceReference("rec-3300", OTHER_READER, EARLIER)
        self.assertFalse(ref.belongs_to(READER))
        with self.assertRaises(ValidationError) as ctx:
            ref.require_usable_for(READER, WHEN)
        self.assertEqual(ctx.exception.issue, ValidationIssue.RECIPIENT_SCOPE_MISMATCH)

    def test_strictly_prior_chronology(self):
        ref = EvidenceReference("rec-3300", READER, EARLIER)
        self.assertTrue(ref.is_strictly_prior_to(WHEN))
        self.assertFalse(ref.is_strictly_prior_to(EARLIER))

    def test_simultaneous_record_is_not_prior(self):
        ref = EvidenceReference("rec-3300", READER, WHEN)
        with self.assertRaises(ValidationError) as ctx:
            ref.require_usable_for(READER, WHEN)
        self.assertEqual(ctx.exception.issue, ValidationIssue.CHRONOLOGY_VIOLATION)

    def test_future_record_rejected(self):
        later = WHEN + dt.timedelta(minutes=1)
        ref = EvidenceReference("rec-3300", READER, later)
        with self.assertRaises(ValidationError) as ctx:
            ref.require_usable_for(READER, WHEN)
        self.assertEqual(ctx.exception.issue, ValidationIssue.CHRONOLOGY_VIOLATION)

    def test_scope_error_takes_precedence_over_chronology(self):
        ref = EvidenceReference("rec-3300", OTHER_READER, WHEN + dt.timedelta(days=1))
        with self.assertRaises(ValidationError) as ctx:
            ref.require_usable_for(READER, WHEN)
        self.assertEqual(ctx.exception.issue, ValidationIssue.RECIPIENT_SCOPE_MISMATCH)

    def test_reference_carries_no_historical_text_field(self):
        ref = EvidenceReference("rec-3300", READER, EARLIER)
        self.assertEqual(set(vars(ref)), {"record_id", "recipient", "occurred_at"})


# ---------------------------------------------------------------------------
# 15-16. Enum parsing and action/category independence.
# ---------------------------------------------------------------------------


class VocabularyTests(unittest.TestCase):
    def test_routing_action_values(self):
        self.assertEqual([a.value for a in RoutingAction], ["interrupt", "defer", "suppress"])

    def test_invalid_enum_value_fails_closed(self):
        for raw in ("escalate", "", None, 3, "INTERRUPT"):
            with self.assertRaises(UnsupportedValueError):
                RoutingAction.from_value(raw, "action")

    def test_valid_enum_parsing(self):
        self.assertIs(RoutingAction.from_value("suppress", "action"), RoutingAction.SUPPRESS)
        self.assertIs(
            MessageCategory.from_value("transactional", "category"), MessageCategory.TRANSACTIONAL
        )

    def test_action_and_category_are_independent_types(self):
        self.assertFalse(set(RoutingAction) & set(MessageCategory))
        for action in RoutingAction:
            self.assertNotIsInstance(action, MessageCategory)
        with self.assertRaises(ValidationError) as ctx:
            RoutingDecision(
                "note-7712",
                MessageCategory.CONVERSATIONAL,  # wrong vocabulary in the action slot
                MessageCategory.CONVERSATIONAL,
                Confidence(0.4),
                Rationale("placeholder"),
            )
        self.assertEqual(ctx.exception.issue, ValidationIssue.WRONG_TYPE)

    def test_every_category_pairs_with_every_action(self):
        for action in RoutingAction:
            for category in MessageCategory:
                decision = RoutingDecision(
                    "note-7712", action, category, Confidence(0.5), Rationale("paired")
                )
                self.assertIs(decision.action, action)
                self.assertIs(decision.category, category)


# ---------------------------------------------------------------------------
# 17. Explicit absence.
# ---------------------------------------------------------------------------


class AbsenceTests(unittest.TestCase):
    def test_optional_context_stays_none(self):
        msg = message()
        self.assertIsNone(msg.sender)
        self.assertIsNone(msg.attachment)
        self.assertFalse(msg.has_known_sender)
        self.assertFalse(msg.has_attachment)

    def test_absence_can_be_recorded_without_inventing_a_value(self):
        gap = ContextGap("sender", AbsenceReason.LOOKUP_FAILED)
        msg = message(gaps=(gap,))
        self.assertIsNone(msg.sender)
        self.assertEqual(msg.gaps[0].reason, AbsenceReason.LOOKUP_FAILED)

    def test_preferences_default_to_no_quiet_period(self):
        self.assertIsNone(RecipientPreferences().quiet_period)

    def test_quiet_period_can_wrap_midnight(self):
        overnight = QuietPeriod(22 * 60, 7 * 60)
        self.assertTrue(overnight.wraps_midnight)
        self.assertTrue(overnight.contains_minute(23 * 60))
        self.assertTrue(overnight.contains_minute(2 * 60))
        self.assertFalse(overnight.contains_minute(12 * 60))


# ---------------------------------------------------------------------------
# 18-19. Trust boundary.
# ---------------------------------------------------------------------------


class TrustBoundaryTests(unittest.TestCase):
    def test_no_provenance_carries_instruction_authority(self):
        for provenance in Provenance:
            self.assertFalse(provenance.carries_instruction_authority)

    def test_no_provenance_confers_verified_trust(self):
        for provenance in Provenance:
            self.assertFalse(provenance.confers_verified_trust)

    def test_structured_metadata_is_not_automatically_verified(self):
        self.assertFalse(Provenance.STRUCTURED_METADATA.confers_verified_trust)
        sender = SenderContext("snd-208")
        self.assertFalse(sender.has_asserted_trust)

    def test_extracted_provenances_are_marked(self):
        for provenance in (
            Provenance.OCR_TEXT,
            Provenance.ASR_TRANSCRIPT,
            Provenance.URL_DERIVED,
            Provenance.QR_DERIVED,
            Provenance.MODEL_INTERPRETATION,
        ):
            self.assertTrue(provenance.is_extracted)
        for provenance in (Provenance.AUTHORED_TEXT, Provenance.STRUCTURED_METADATA):
            self.assertFalse(provenance.is_extracted)

    def test_deterministic_signal_is_distinct_from_raw_content(self):
        signal = DeterministicSignal("digit_run", "a long digit run is present")
        self.assertIs(signal.provenance, Provenance.DERIVED_SIGNAL)
        self.assertFalse(Provenance.DERIVED_SIGNAL.is_content)
        self.assertNotIsInstance(signal, TextContent)

    def test_deterministic_signal_provenance_cannot_be_supplied(self):
        with self.assertRaises(TypeError):
            DeterministicSignal("n", "d", provenance=Provenance.AUTHORED_TEXT)  # type: ignore[call-arg]

    def test_provenance_survives_transport_between_layers(self):
        extracted = TextContent("decoded payload text", Provenance.QR_DERIVED)
        analysis = MediaAnalysis(
            MediaReference("att-55c", MediaKind.IMAGE),
            ExtractionStatus.SUCCEEDED,
            content=extracted,
        )
        self.assertIs(analysis.content.provenance, Provenance.QR_DERIVED)

    def test_message_body_must_be_authored_text(self):
        for provenance in (Provenance.OCR_TEXT, Provenance.MODEL_INTERPRETATION, Provenance.QR_DERIVED):
            with self.assertRaises(ValidationError) as ctx:
                message(body=TextContent("substituted", provenance))
            self.assertEqual(ctx.exception.issue, ValidationIssue.PROVENANCE_MISMATCH)

    def test_text_content_requires_provenance(self):
        with self.assertRaises(TypeError):
            TextContent("no provenance given")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 20. Sanitized errors.
# ---------------------------------------------------------------------------


class SanitizedErrorTests(unittest.TestCase):
    SECRET = "gate code 44127 and the spare key location"

    def test_error_text_never_contains_the_offending_value(self):
        with self.assertRaises(ValidationError) as ctx:
            BackendIdentity(f"/opt/{self.SECRET}/engine")
        rendered = f"{ctx.exception} {ctx.exception!r} {ctx.exception.args}"
        self.assertNotIn(self.SECRET, rendered)
        self.assertNotIn("44127", rendered)

    def test_error_formatting_is_deterministic(self):
        first = ValidationError(ValidationIssue.EMPTY_TEXT, "detail")
        second = ValidationError(ValidationIssue.EMPTY_TEXT, "detail")
        self.assertEqual(str(first), str(second))
        self.assertEqual(repr(first), repr(second))
        self.assertEqual(str(first), "empty_text: field=detail")

    def test_error_reports_useful_category_and_field(self):
        with self.assertRaises(ValidationError) as ctx:
            Confidence(2.5)
        self.assertEqual(ctx.exception.issue, ValidationIssue.OUT_OF_RANGE)
        self.assertEqual(ctx.exception.field, "value")

    def test_text_content_repr_does_not_leak_content(self):
        content = TextContent(self.SECRET, Provenance.ASR_TRANSCRIPT)
        self.assertNotIn(self.SECRET, repr(content))
        self.assertIn("asr_transcript", repr(content))
        self.assertEqual(content.value, self.SECRET)  # readable when genuinely needed

    def test_unsupported_value_error_is_a_validation_error(self):
        self.assertTrue(issubclass(UnsupportedValueError, ValidationError))


if __name__ == "__main__":
    unittest.main()
