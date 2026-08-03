"""The public request-document format.

One document describes **one** routing request. The shape is designed for
this project rather than adapted from any other file layout: a single
nested object, no tables, no row relationships, no cross-file joins.

Example of the whole format::

    {
      "format": "message-router.request",
      "version": 1,
      "request": {
        "recipient": {
          "id": "rcp-4kq",
          "preferences": {
            "quiet_from_minute": 1320,
            "quiet_to_minute": 420,
            "silenced_conversations": ["conv-31b"],
            "accepts_broadcast": true
          }
        },
        "conversation": {"id": "conv-31b", "surface": "direct", "participants": 2},
        "sender": {"id": "snd-208"},
        "message": {
          "id": "note-7712",
          "sent_at": "2031-03-09T14:30:00+00:00",
          "body": "text a person typed",
          "attachment": {"id": "att-55c", "kind": "image", "source": "notices/gate.png"}
        },
        "history": [
          {"id": "rec-3300", "occurred_at": "2031-03-09T09:05:00+00:00",
           "conversation_id": "conv-31b", "opened": true, "replied": false,
           "dismissed": false}
        ]
      }
    }

Rules the format commits to:

* **Versioned.** ``format`` and ``version`` are both required, and an
  unrecognised value for either is refused rather than guessed at.
* **Unknown fields are rejected.** This is the documented policy, chosen
  over silently ignoring them: a misspelled key is far more often a
  mistake the author wants to hear about than a deliberate extension, and
  ignoring it would apply a default the author never asked for. Adding a
  field is therefore a version change.
* **Inputs only.** There is no field for an action, category, confidence,
  reason, risk, trust, score, or label. A document describes what arrived;
  it cannot dictate, pre-empt, or override what the system decides about
  it.
* **No fabrication.** Anything optional that is absent stays absent. The
  loader never substitutes a default sender, an empty body, or a
  timestamp.
"""

from __future__ import annotations

__all__ = [
    "FORMAT_NAME",
    "SUPPORTED_VERSIONS",
    "UNKNOWN_FIELD_POLICY",
    "DOCUMENT_FIELDS",
    "REQUEST_FIELDS",
    "MESSAGE_FIELDS",
    "ATTACHMENT_FIELDS",
    "RECIPIENT_FIELDS",
    "PREFERENCE_FIELDS",
    "CONVERSATION_FIELDS",
    "SENDER_FIELDS",
    "HISTORY_FIELDS",
]

#: Identifies documents intended for this loader.
FORMAT_NAME = "message-router.request"

#: Versions this build accepts. Anything else fails closed.
SUPPORTED_VERSIONS = frozenset({1})

#: The single, documented policy for keys the format does not define.
UNKNOWN_FIELD_POLICY = "reject"

DOCUMENT_FIELDS = frozenset({"format", "version", "request"})
REQUEST_FIELDS = frozenset({"recipient", "conversation", "sender", "message", "history"})
MESSAGE_FIELDS = frozenset({"id", "sent_at", "body", "attachment"})
ATTACHMENT_FIELDS = frozenset({"id", "kind", "source"})
RECIPIENT_FIELDS = frozenset({"id", "preferences"})
PREFERENCE_FIELDS = frozenset(
    {"quiet_from_minute", "quiet_to_minute", "silenced_conversations", "accepts_broadcast"}
)
CONVERSATION_FIELDS = frozenset({"id", "surface", "participants"})
SENDER_FIELDS = frozenset({"id"})
HISTORY_FIELDS = frozenset({"id", "occurred_at", "conversation_id", "opened", "replied", "dismissed"})
