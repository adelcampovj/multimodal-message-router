"""Tests for recipient-scoped history access."""

from __future__ import annotations

import datetime as dt
import inspect
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from message_router.domain import (  # noqa: E402
    EvidenceReference,
    RecipientScope,
    ValidationError,
    ValidationIssue,
)
from message_router.ports import EvidenceRepository  # noqa: E402
from message_router.repositories import InMemoryHistoryRepository  # noqa: E402
from message_router.repositories import history as history_module  # noqa: E402

READER = RecipientScope("rcp-4kq")
OTHER = RecipientScope("rcp-9wz")
CUTOFF = dt.datetime(2031, 3, 9, 14, 30, tzinfo=dt.timezone.utc)


def ref(identifier: str, recipient: RecipientScope, minutes_before: int) -> EvidenceReference:
    return EvidenceReference(identifier, recipient, CUTOFF - dt.timedelta(minutes=minutes_before))


class ScopingTests(unittest.TestCase):
    def test_returns_only_the_requested_recipients_records(self):
        repo = InMemoryHistoryRepository(
            [ref("rec-1", READER, 60), ref("rec-2", OTHER, 60), ref("rec-3", READER, 30)]
        )
        returned = repo.references_before(READER, CUTOFF, 10)
        self.assertEqual([r.record_id for r in returned], ["rec-3", "rec-1"])
        for reference in returned:
            self.assertTrue(reference.belongs_to(READER))

    def test_cross_recipient_records_excluded(self):
        repo = InMemoryHistoryRepository([ref("rec-2", OTHER, 60)])
        self.assertEqual(repo.references_before(READER, CUTOFF, 10), ())

    def test_returned_records_satisfy_the_domain_usability_check(self):
        repo = InMemoryHistoryRepository([ref("rec-1", READER, 60)])
        for reference in repo.references_before(READER, CUTOFF, 10):
            reference.require_usable_for(READER, CUTOFF)  # must not raise


class ChronologyTests(unittest.TestCase):
    def test_only_strictly_earlier_records_returned(self):
        repo = InMemoryHistoryRepository(
            [ref("rec-past", READER, 60), ref("rec-future", READER, -60)]
        )
        returned = repo.references_before(READER, CUTOFF, 10)
        self.assertEqual([r.record_id for r in returned], ["rec-past"])

    def test_record_at_exactly_the_cutoff_is_excluded(self):
        repo = InMemoryHistoryRepository([ref("rec-now", READER, 0)])
        self.assertEqual(repo.references_before(READER, CUTOFF, 10), ())

    def test_cutoff_may_be_supplied_in_another_offset(self):
        east = dt.timezone(dt.timedelta(hours=4))
        repo = InMemoryHistoryRepository([ref("rec-1", READER, 60)])
        returned = repo.references_before(READER, CUTOFF.astimezone(east), 10)
        self.assertEqual([r.record_id for r in returned], ["rec-1"])

    def test_naive_cutoff_rejected(self):
        repo = InMemoryHistoryRepository([ref("rec-1", READER, 60)])
        with self.assertRaises(ValidationError) as ctx:
            repo.references_before(READER, dt.datetime(2031, 3, 9, 14, 30), 10)
        self.assertEqual(ctx.exception.issue, ValidationIssue.NAIVE_TIMESTAMP)


class OrderingAndLimitTests(unittest.TestCase):
    def test_ordering_is_newest_first_then_by_identity(self):
        repo = InMemoryHistoryRepository(
            [ref("rec-c", READER, 10), ref("rec-a", READER, 90), ref("rec-b", READER, 50)]
        )
        self.assertEqual(
            [r.record_id for r in repo.references_before(READER, CUTOFF, 10)],
            ["rec-c", "rec-b", "rec-a"],
        )

    def test_ties_broken_deterministically_by_identity(self):
        repo = InMemoryHistoryRepository(
            [ref("rec-z", READER, 30), ref("rec-a", READER, 30), ref("rec-m", READER, 30)]
        )
        self.assertEqual(
            [r.record_id for r in repo.references_before(READER, CUTOFF, 10)],
            ["rec-a", "rec-m", "rec-z"],
        )

    def test_insertion_order_does_not_affect_results(self):
        entries = [ref("rec-a", READER, 90), ref("rec-b", READER, 50), ref("rec-c", READER, 10)]
        forward = InMemoryHistoryRepository(entries).references_before(READER, CUTOFF, 10)
        backward = InMemoryHistoryRepository(list(reversed(entries))).references_before(READER, CUTOFF, 10)
        self.assertEqual([r.record_id for r in forward], [r.record_id for r in backward])

    def test_limit_applied_after_ordering(self):
        repo = InMemoryHistoryRepository(
            [ref("rec-a", READER, 90), ref("rec-b", READER, 50), ref("rec-c", READER, 10)]
        )
        self.assertEqual([r.record_id for r in repo.references_before(READER, CUTOFF, 2)], ["rec-c", "rec-b"])
        self.assertEqual(repo.references_before(READER, CUTOFF, 0), ())

    def test_invalid_limit_rejected(self):
        repo = InMemoryHistoryRepository([])
        for bad in (-1, True, "2", 1.5):
            with self.assertRaises(ValidationError):
                repo.references_before(READER, CUTOFF, bad)


class ConstructionTests(unittest.TestCase):
    def test_duplicate_record_identity_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            InMemoryHistoryRepository([ref("rec-1", READER, 60), ref("rec-1", READER, 30)])
        self.assertEqual(ctx.exception.issue, ValidationIssue.DUPLICATE_IDENTIFIER)

    def test_duplicate_identity_rejected_even_across_recipients(self):
        with self.assertRaises(ValidationError):
            InMemoryHistoryRepository([ref("rec-1", READER, 60), ref("rec-1", OTHER, 30)])

    def test_non_reference_entry_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            InMemoryHistoryRepository(["rec-1"])
        self.assertEqual(ctx.exception.issue, ValidationIssue.WRONG_TYPE)

    def test_empty_repository_is_valid(self):
        self.assertEqual(len(InMemoryHistoryRepository()), 0)


class PortSubstitutabilityTests(unittest.TestCase):
    def test_repository_satisfies_the_evidence_port(self):
        self.assertIsInstance(InMemoryHistoryRepository(), EvidenceRepository)

    def test_repository_subclasses_nothing_from_the_ports_package(self):
        self.assertEqual(InMemoryHistoryRepository.__bases__, (object,))

    def test_a_substitute_double_satisfies_the_same_port(self):
        class StubRepository:
            def references_before(self, recipient, moment, limit):
                return ()

        self.assertIsInstance(StubRepository(), EvidenceRepository)


class NoUnscopedApiTests(unittest.TestCase):
    def test_the_only_public_query_requires_recipient_and_cutoff(self):
        public = [
            name
            for name, _ in inspect.getmembers(InMemoryHistoryRepository, inspect.isfunction)
            if not name.startswith("_")
        ]
        self.assertEqual(public, ["references_before"])
        signature = inspect.signature(InMemoryHistoryRepository.references_before)
        self.assertEqual(list(signature.parameters), ["self", "recipient", "moment", "limit"])
        for required in ("recipient", "moment"):
            self.assertIs(signature.parameters[required].default, inspect.Parameter.empty)

    def test_module_exposes_no_unscoped_helper(self):
        for name in dir(history_module):
            if name.startswith("_"):
                continue
            lowered = name.lower()
            for banned in ("all_", "everything", "unscoped", "global"):
                self.assertNotIn(banned, lowered)

    def test_module_implements_no_ranking_or_scoring(self):
        """Scans identifiers, not raw text.

        A substring search over the whole file would match this module's own
        docstring, which promises it does *not* rank or score -- prose that
        names a thing is not an implementation of it.
        """
        import ast

        source = (SRC_DIR / "message_router" / "repositories" / "history.py").read_text(encoding="utf-8")
        identifiers: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                identifiers.add(node.name)
            elif isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
            elif isinstance(node, ast.arg):
                identifiers.add(node.arg)

        banned = ("score", "rank", "relevance", "similarity", "tfidf", "jaccard", "weight")
        offenders = [name for name in identifiers if any(term in name.lower() for term in banned)]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
