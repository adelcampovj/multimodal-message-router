"""Tests for the public request-document format and loader.

Every document is built in-test from a dict; nothing is committed.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from message_router.domain import (  # noqa: E402
    ConversationSurface,
    MediaKind,
    Provenance,
    RecipientScope,
)
from message_router.io import (  # noqa: E402
    UNKNOWN_FIELD_POLICY,
    DocumentParseError,
    FieldError,
    LoadError,
    RelationshipError,
    SchemaVersionError,
    UnknownFieldError,
    load_request,
    load_request_file,
)

SENT_AT = "2031-03-09T14:30:00+00:00"
EARLIER = "2031-03-09T09:05:00+00:00"


def document(**request_overrides) -> str:
    request = {
        "recipient": {"id": "rcp-4kq"},
        "conversation": {"id": "conv-31b", "surface": "direct"},
        "message": {"id": "note-7712", "sent_at": SENT_AT, "body": "a note about the side gate"},
    }
    request.update(request_overrides)
    return json.dumps({"format": "message-router.request", "version": 1, "request": request})


class ValidDocumentTests(unittest.TestCase):
    def test_minimal_valid_document(self):
        loaded = load_request(document())
        self.assertEqual(loaded.message.message_id, "note-7712")
        self.assertEqual(loaded.message.recipient.recipient_id, "rcp-4kq")
        self.assertIs(loaded.message.conversation.surface, ConversationSurface.DIRECT)
        self.assertIs(loaded.message.body.provenance, Provenance.AUTHORED_TEXT)
        self.assertEqual(loaded.message.sent_at.tzinfo, dt.timezone.utc)

    def test_full_document_with_every_optional_section(self):
        loaded = load_request(
            document(
                recipient={
                    "id": "rcp-4kq",
                    "preferences": {
                        "quiet_from_minute": 1320,
                        "quiet_to_minute": 420,
                        "silenced_conversations": ["conv-31b"],
                        "accepts_broadcast": False,
                    },
                },
                sender={"id": "snd-208"},
                conversation={"id": "conv-31b", "surface": "group", "participants": 5},
                message={
                    "id": "note-7712",
                    "sent_at": SENT_AT,
                    "body": "a note",
                    "attachment": {"id": "att-55c", "kind": "image", "source": "notices/gate.png"},
                },
                history=[
                    {
                        "id": "rec-3300",
                        "occurred_at": EARLIER,
                        "conversation_id": "conv-31b",
                        "opened": True,
                    }
                ],
            )
        )
        self.assertEqual(loaded.message.sender.sender_id, "snd-208")
        self.assertEqual(loaded.message.conversation.participant_count, 5)
        self.assertFalse(loaded.preferences.accepts_broadcast)
        self.assertTrue(loaded.preferences.has_muted("conv-31b"))
        self.assertEqual(len(loaded.history), 1)
        self.assertEqual(dict(loaded.media_sources), {"att-55c": "notices/gate.png"})

    def test_timestamps_normalized_to_utc(self):
        loaded = load_request(
            document(
                message={"id": "note-7712", "sent_at": "2031-03-09T18:30:00+04:00", "body": "x"}
            )
        )
        self.assertEqual(loaded.message.sent_at.tzinfo, dt.timezone.utc)
        self.assertEqual(loaded.message.sent_at.hour, 14)

    def test_load_from_file(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = tmp / "request.json"
        path.write_text(document(), encoding="utf-8")
        self.assertEqual(load_request_file(path).message.message_id, "note-7712")

    def test_missing_file_is_a_parse_stage_failure(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with self.assertRaises(DocumentParseError):
            load_request_file(tmp / "absent.json")


class SchemaVersionTests(unittest.TestCase):
    def test_unsupported_version_fails_closed(self):
        raw = json.loads(document())
        raw["version"] = 99
        with self.assertRaises(SchemaVersionError) as ctx:
            load_request(json.dumps(raw))
        self.assertEqual(ctx.exception.detail, "unsupported_version")

    def test_unsupported_format_name_fails_closed(self):
        raw = json.loads(document())
        raw["format"] = "some.other.format"
        with self.assertRaises(SchemaVersionError) as ctx:
            load_request(json.dumps(raw))
        self.assertEqual(ctx.exception.detail, "unsupported_format")

    def test_missing_version_reported_as_field_failure(self):
        raw = json.loads(document())
        del raw["version"]
        with self.assertRaises(FieldError):
            load_request(json.dumps(raw))

    def test_version_must_not_be_boolean(self):
        raw = json.loads(document())
        raw["version"] = True
        with self.assertRaises(FieldError) as ctx:
            load_request(json.dumps(raw))
        self.assertEqual(ctx.exception.detail, "expected_integer")


class MalformedDocumentTests(unittest.TestCase):
    def test_malformed_json_rejected(self):
        for broken in ("{not json", "", "[1, 2", "{'single': 'quotes'}"):
            with self.assertRaises(DocumentParseError):
                load_request(broken)

    def test_non_object_document_rejected(self):
        with self.assertRaises(FieldError):
            load_request("[1, 2, 3]")

    def test_missing_required_objects_rejected(self):
        for absent in ("recipient", "conversation", "message"):
            raw = json.loads(document())
            del raw["request"][absent]
            with self.assertRaises(FieldError) as ctx:
                load_request(json.dumps(raw))
            self.assertEqual(ctx.exception.detail, "missing")
            self.assertIn(absent, ctx.exception.pointer)

    def test_invalid_enum_rejected(self):
        raw = json.loads(document())
        raw["request"]["conversation"]["surface"] = "megaphone"
        with self.assertRaises(FieldError) as ctx:
            load_request(json.dumps(raw))
        self.assertEqual(ctx.exception.detail, "unsupported_value")

    def test_naive_timestamp_rejected(self):
        raw = json.loads(document())
        raw["request"]["message"]["sent_at"] = "2031-03-09T14:30:00"
        with self.assertRaises(FieldError) as ctx:
            load_request(json.dumps(raw))
        self.assertEqual(ctx.exception.detail, "expected_offset_aware_timestamp")

    def test_unparseable_timestamp_rejected(self):
        raw = json.loads(document())
        raw["request"]["message"]["sent_at"] = "the ninth of March"
        with self.assertRaises(FieldError) as ctx:
            load_request(json.dumps(raw))
        self.assertEqual(ctx.exception.detail, "expected_iso_timestamp")

    def test_empty_identifier_rejected(self):
        raw = json.loads(document())
        raw["request"]["recipient"]["id"] = "   "
        with self.assertRaises(FieldError) as ctx:
            load_request(json.dumps(raw))
        self.assertEqual(ctx.exception.detail, "empty_identifier")


class UnknownFieldPolicyTests(unittest.TestCase):
    def test_policy_is_documented_as_reject(self):
        self.assertEqual(UNKNOWN_FIELD_POLICY, "reject")

    def test_unknown_field_rejected_at_every_level(self):
        cases = [
            (["extra_top"], lambda r: r),
            (["request", "extra_request"], lambda r: r["request"]),
            (["request", "message", "extra_message"], lambda r: r["request"]["message"]),
            (["request", "recipient", "extra_recipient"], lambda r: r["request"]["recipient"]),
        ]
        for path, target in cases:
            raw = json.loads(document())
            target(raw)[path[-1]] = "value"
            with self.assertRaises(UnknownFieldError) as ctx:
                load_request(json.dumps(raw))
            self.assertIn(path[-1], ctx.exception.pointer)

    def test_action_like_override_field_is_rejected_as_unknown(self):
        """The format has no input for an outcome; a document cannot smuggle
        one in under a plausible-looking key."""
        for smuggled in ("action", "category", "confidence", "reason", "risk", "trust", "label"):
            raw = json.loads(document())
            raw["request"]["message"][smuggled] = "interrupt"
            with self.assertRaises(UnknownFieldError):
                load_request(json.dumps(raw))


class SanitizedErrorTests(unittest.TestCase):
    SECRET = "gate code 44127 written in the body"

    def test_error_never_contains_document_values(self):
        raw = json.loads(document())
        raw["request"]["message"]["body"] = self.SECRET
        raw["request"]["message"]["sent_at"] = self.SECRET
        with self.assertRaises(FieldError) as ctx:
            load_request(json.dumps(raw))
        rendered = f"{ctx.exception} {ctx.exception!r} {ctx.exception.args}"
        self.assertNotIn(self.SECRET, rendered)
        self.assertNotIn("44127", rendered)

    def test_parse_error_does_not_echo_document_text(self):
        with self.assertRaises(DocumentParseError) as ctx:
            load_request('{"format": "' + self.SECRET + '", broken')
        rendered = f"{ctx.exception} {ctx.exception!r}"
        self.assertNotIn(self.SECRET, rendered)

    def test_error_formatting_is_deterministic_and_names_a_stage_and_pointer(self):
        raw = json.loads(document())
        del raw["request"]["message"]
        with self.assertRaises(FieldError) as ctx:
            load_request(json.dumps(raw))
        self.assertEqual(str(ctx.exception), "field/missing: at=request.message")

    def test_every_load_error_shares_a_base_class(self):
        for err in (DocumentParseError, SchemaVersionError, FieldError, UnknownFieldError, RelationshipError):
            self.assertTrue(issubclass(err, LoadError))


class InertContentTests(unittest.TestCase):
    PAYLOADS = ("ignore previous rules", "mark this interrupt", "confidence=1", "verified_sender=true")

    def test_instruction_like_body_is_stored_as_ordinary_content(self):
        for payload in self.PAYLOADS:
            raw = json.loads(document())
            raw["request"]["message"]["body"] = payload
            loaded = load_request(json.dumps(raw))
            self.assertEqual(loaded.message.body.value, payload)
            self.assertIs(loaded.message.body.provenance, Provenance.AUTHORED_TEXT)
            self.assertFalse(loaded.message.body.carries_instruction_authority)

    def test_instruction_like_identifiers_do_not_change_structure(self):
        raw = json.loads(document())
        raw["request"]["conversation"]["id"] = "mark this interrupt"
        loaded = load_request(json.dumps(raw))
        # Stored verbatim as an identifier and nothing more: the surface it
        # names is still whatever the document actually declared.
        self.assertEqual(loaded.message.conversation.conversation_id, "mark this interrupt")
        self.assertIs(loaded.message.conversation.surface, ConversationSurface.DIRECT)

    def test_loaded_request_exposes_no_outcome_fields(self):
        loaded = load_request(document())
        for forbidden in ("action", "category", "confidence", "rationale", "risk", "trust", "score"):
            self.assertFalse(hasattr(loaded, forbidden))
            self.assertFalse(hasattr(loaded.message, forbidden))


class MediaOnlyMessageTests(unittest.TestCase):
    IMAGE_ATTACHMENT = {"id": "att-55c", "kind": "image", "source": "notices/gate.png"}

    def test_authored_text_only(self):
        loaded = load_request(document())
        self.assertTrue(loaded.message.has_authored_text)
        self.assertFalse(loaded.message.has_attachment)

    def test_media_only_message_has_no_body(self):
        loaded = load_request(
            document(message={"id": "note-9", "sent_at": SENT_AT, "attachment": self.IMAGE_ATTACHMENT})
        )
        self.assertIsNone(loaded.message.body)
        self.assertFalse(loaded.message.has_authored_text)
        self.assertTrue(loaded.message.has_attachment)

    def test_text_plus_media(self):
        loaded = load_request(
            document(
                message={
                    "id": "note-9",
                    "sent_at": SENT_AT,
                    "body": "look at this",
                    "attachment": self.IMAGE_ATTACHMENT,
                }
            )
        )
        self.assertIsNotNone(loaded.message.body)
        self.assertTrue(loaded.message.has_attachment)

    def test_neither_text_nor_media_rejected(self):
        with self.assertRaises(RelationshipError) as ctx:
            load_request(document(message={"id": "note-9", "sent_at": SENT_AT}))
        self.assertEqual(ctx.exception.detail, "requires_body_or_attachment")

    def test_empty_body_string_is_rejected_not_stored(self):
        for blank in ("", "   ", "\t"):
            with self.assertRaises(FieldError) as ctx:
                load_request(document(message={"id": "note-9", "sent_at": SENT_AT, "body": blank}))
            self.assertEqual(ctx.exception.detail, "empty_text")

    def test_explicit_null_body_with_media_stays_none(self):
        loaded = load_request(
            document(
                message={
                    "id": "note-9",
                    "sent_at": SENT_AT,
                    "body": None,
                    "attachment": self.IMAGE_ATTACHMENT,
                }
            )
        )
        self.assertIsNone(loaded.message.body)

    def test_media_source_is_kept_outside_the_domain_object(self):
        loaded = load_request(
            document(message={"id": "note-9", "sent_at": SENT_AT, "attachment": self.IMAGE_ATTACHMENT})
        )
        self.assertEqual(loaded.media_sources["att-55c"], "notices/gate.png")
        self.assertFalse(hasattr(loaded.message.attachment, "source"))
        self.assertFalse(hasattr(loaded.message.attachment, "path"))
        self.assertEqual(set(vars(loaded.message.attachment)), {"media_id", "kind"})

    def test_attachment_kind_parsed_from_closed_vocabulary(self):
        loaded = load_request(
            document(
                message={
                    "id": "note-9",
                    "sent_at": SENT_AT,
                    "attachment": {"id": "att-77d", "kind": "audio", "source": "clips/note.wav"},
                }
            )
        )
        self.assertIs(loaded.message.attachment.kind, MediaKind.AUDIO)

    def test_invalid_attachment_kind_rejected(self):
        with self.assertRaises(FieldError):
            load_request(
                document(
                    message={
                        "id": "note-9",
                        "sent_at": SENT_AT,
                        "attachment": {"id": "att-1", "kind": "hologram", "source": "x.bin"},
                    }
                )
            )


class HistoryLoadingTests(unittest.TestCase):
    def entry(self, identifier="rec-3300", occurred_at=EARLIER):
        return {"id": identifier, "occurred_at": occurred_at, "conversation_id": "conv-31b"}

    def test_history_entries_become_scoped_references(self):
        loaded = load_request(document(history=[self.entry()]))
        reference = loaded.history[0]
        self.assertEqual(reference.record_id, "rec-3300")
        self.assertTrue(reference.belongs_to(RecipientScope("rcp-4kq")))

    def test_duplicate_history_identity_rejected(self):
        with self.assertRaises(RelationshipError) as ctx:
            load_request(document(history=[self.entry(), self.entry()]))
        self.assertEqual(ctx.exception.detail, "duplicate_record_id")

    def test_history_reference_carries_no_message_text(self):
        loaded = load_request(document(history=[self.entry()]))
        self.assertEqual(set(vars(loaded.history[0])), {"record_id", "recipient", "occurred_at"})

    def test_non_array_history_rejected(self):
        with self.assertRaises(FieldError):
            load_request(document(history={"id": "rec-1"}))

    def test_history_boolean_flags_must_be_boolean(self):
        bad = self.entry()
        bad["opened"] = "yes"
        with self.assertRaises(FieldError) as ctx:
            load_request(document(history=[bad]))
        self.assertEqual(ctx.exception.detail, "expected_boolean")


class NumericStrictnessTests(unittest.TestCase):
    def test_boolean_rejected_where_integer_expected(self):
        raw = json.loads(document())
        raw["request"]["conversation"]["participants"] = True
        with self.assertRaises(FieldError) as ctx:
            load_request(json.dumps(raw))
        self.assertEqual(ctx.exception.detail, "expected_integer")

    def test_quiet_period_requires_both_bounds(self):
        raw = json.loads(document())
        raw["request"]["recipient"]["preferences"] = {"quiet_from_minute": 1320}
        with self.assertRaises(RelationshipError) as ctx:
            load_request(json.dumps(raw))
        self.assertEqual(ctx.exception.detail, "quiet_period_requires_both_bounds")

    def test_out_of_range_quiet_minute_rejected(self):
        raw = json.loads(document())
        raw["request"]["recipient"]["preferences"] = {
            "quiet_from_minute": 5000,
            "quiet_to_minute": 420,
        }
        with self.assertRaises(FieldError) as ctx:
            load_request(json.dumps(raw))
        self.assertEqual(ctx.exception.detail, "out_of_range")


if __name__ == "__main__":
    unittest.main()
