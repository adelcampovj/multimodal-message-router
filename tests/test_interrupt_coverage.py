"""An interrupt must rest on content the safety layer actually analysed.

The safety layer analyses at most ``MAX_ANALYZED_CHARS`` characters of each
source. Classification still reads every source in full, so before this guard
a message padded past the limit could carry alert wording in its unread tail,
be classified as a safety alert, and interrupt -- even when the same tail
also held a credential request that, read, would have been vetoed.

These tests pin the guard, and pin that it is narrow:

* urgency found only past the limit -- in any source -- cannot earn an
  interrupt, and structured context cannot amplify it into one;
* urgency found inside the analysed part still interrupts, truncated or not;
* a severe finding inside the analysed part still vetoes first;
* nothing past the limit is claimed as detected, and ``analysis_incomplete``
  stays true;
* classification, findings and every other route are unchanged.

All message text is fictional and written for these tests.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from message_router.analysis import classify, extract_features  # noqa: E402
from message_router.analysis.features import FeatureFamily, FeatureLevel, SupportSource  # noqa: E402
from message_router.domain import (  # noqa: E402
    ExtractionStatus,
    MediaAnalysis,
    Provenance,
    TextContent,
)
from message_router.domain.media import BackendIdentity  # noqa: E402
from message_router.io import load_request  # noqa: E402
from message_router.media import MediaCoordinator  # noqa: E402
from message_router.pipeline import RoutedRequest, decision_to_dict, route_request  # noqa: E402
from message_router.routing import route  # noqa: E402
from message_router.safety import analyze_message  # noqa: E402
from message_router.safety.normalization import MAX_ANALYZED_CHARS as CAP  # noqa: E402

NEUTRAL = "The garden club planted tulips near the old stone wall and talked about compost. "
ALERT = "Fire alarm in the east wing. Please evacuate by the north stairs now."
CREDENTIAL = (
    "Hello, this is the helpdesk. Please send me the one time code we just texted you "
    "so we can verify your account."
)
DISRUPTION = "The payment portal is currently down for all users."
APPOINTMENT = "Your dentist appointment is at 3 pm."
#: Assembled so the marker is not a literal anywhere a search would match it.
MARKER = "QZX" + "TAILMARK" + "7731"

INTERRUPT = ("interrupt", "route.urgency.safe_interrupt")
VETO = ("suppress", "route.safety.hard_veto")


def _pad(length: int) -> str:
    """Exactly ``length`` neutral characters, ending on a space."""
    text = (NEUTRAL * (length // len(NEUTRAL) + 2))[: length - 1]
    return text + " "


def _document(body, attachment=None, surface="direct", participants=2, preferences=None) -> str:
    message = {"id": "msg-coverage", "sent_at": "2031-03-09T14:30:00+00:00"}
    if body is not None:
        message["body"] = body
    if attachment is not None:
        message["attachment"] = attachment
    recipient = {"id": "rcp-coverage"}
    if preferences is not None:
        recipient["preferences"] = preferences
    return json.dumps({
        "format": "message-router.request",
        "version": 1,
        "request": {
            "recipient": recipient,
            "conversation": {"id": "conv-coverage", "surface": surface, "participants": participants},
            "sender": {"id": "snd-coverage"},
            "message": message,
        },
    })


class _Fixture:
    """Returns fixed fictional text for one role; reads no media."""

    def __init__(self, text, provenance):
        self.text = text
        self.provenance = provenance

    def _analysis(self, media):
        identity = BackendIdentity("coverage-fixture", "1")
        if self.text is None:
            return MediaAnalysis(media, ExtractionStatus.NO_CONTENT_FOUND, backend=identity)
        return MediaAnalysis(media, ExtractionStatus.SUCCEEDED, backend=identity,
                             content=TextContent(self.text, self.provenance))

    extract_text = transcribe = decode = _analysis


def _routed(body=None, *, kind=None, ocr=None, asr=None, qr=None, **document) -> RoutedRequest:
    """Route through the public pipeline, with fixtures standing in for engines."""
    attachment = coordinator = None
    if kind is not None:
        attachment = {"id": "att-coverage", "kind": kind, "source": "synthetic/not-shipped"}
        coordinator = MediaCoordinator(
            ocr=_Fixture(ocr, Provenance.OCR_TEXT),
            asr=_Fixture(asr, Provenance.ASR_TRANSCRIPT),
            qr=_Fixture(qr, Provenance.QR_DERIVED),
        )
    return route_request(load_request(_document(body, attachment, **document)), coordinator=coordinator)


def _routed_with_context(body, *, directly_addressed=False, near_term_event=False) -> RoutedRequest:
    """The pipeline's stages, plus the structured context the request format cannot carry."""
    loaded = load_request(_document(body))
    assessment = analyze_message(loaded.message)
    classification = classify(((body, Provenance.AUTHORED_TEXT),), assessment,
                              directly_addressed=directly_addressed)
    features = extract_features(loaded.message, assessment, classification,
                                directly_addressed=directly_addressed, near_term_event=near_term_event)
    return RoutedRequest(assessment, classification, features, route(assessment, classification, features))


def _outcome(routed: RoutedRequest) -> tuple[str, str]:
    return routed.result.action.value, routed.result.policy_id


def _rule_ids(routed: RoutedRequest) -> list[str]:
    return [finding.rule_id for finding in routed.assessment.findings]


class LengthBoundaryTests(unittest.TestCase):
    def test_alert_ending_one_character_short_of_the_limit_interrupts(self):
        routed = _routed(_pad(CAP - 1 - len(ALERT)) + ALERT)
        self.assertEqual(_outcome(routed), INTERRUPT)
        self.assertFalse(routed.assessment.analysis_incomplete)

    def test_alert_ending_exactly_at_the_limit_interrupts(self):
        routed = _routed(_pad(CAP - len(ALERT)) + ALERT)
        self.assertEqual(_outcome(routed), INTERRUPT)
        self.assertFalse(routed.assessment.analysis_incomplete)

    def test_alert_losing_only_its_final_full_stop_still_interrupts(self):
        """One character over the limit: the alert's words were all analysed."""
        routed = _routed(_pad(CAP + 1 - len(ALERT)) + ALERT)
        self.assertEqual(_outcome(routed), INTERRUPT)
        self.assertTrue(routed.assessment.analysis_incomplete)

    def test_alert_at_the_start_of_a_far_oversized_message_interrupts(self):
        routed = _routed(ALERT + " " + _pad(3 * CAP))
        self.assertEqual(_outcome(routed), INTERRUPT)
        self.assertTrue(routed.assessment.analysis_incomplete)

    def test_alert_wholly_past_the_limit_does_not_interrupt(self):
        routed = _routed(_pad(CAP) + ALERT)
        self.assertNotEqual(routed.result.action.value, "interrupt")
        self.assertTrue(routed.result.analysis_incomplete)
        self.assertEqual(routed.assessment.findings, ())
        # Classification still describes the whole message; only the
        # interrupt is withheld.
        self.assertEqual(routed.classification.selected.value, "safety_alert")
        self.assertTrue(routed.features.urgency_outside_safety_coverage)

    def test_alert_phrase_split_by_the_limit_does_not_interrupt(self):
        routed = _routed(_pad(CAP - 5) + ALERT + " " + _pad(500))
        self.assertNotEqual(routed.result.action.value, "interrupt")

    def test_alert_phrase_ending_exactly_at_the_limit_counts(self):
        """"Fire alarm" fits inside the analysed part; the rest of the alert does not."""
        routed = _routed(_pad(CAP - len("Fire alarm")) + ALERT + " " + _pad(500))
        self.assertEqual(_outcome(routed), INTERRUPT)
        self.assertFalse(routed.features.urgency_outside_safety_coverage)

    def test_a_word_cut_by_the_limit_is_not_evidence(self):
        """The analysed part ends "evacuat": a match on that half-word is made by the cut."""
        sentence = "Please evacuate the building now."
        cut = sentence.index("evacuate") + len("evacuat")
        routed = _routed(_pad(CAP - cut) + sentence + " " + _pad(500))
        self.assertEqual(routed.classification.selected.value, "safety_alert")
        self.assertNotEqual(routed.result.action.value, "interrupt")


class CredentialPlacementTests(unittest.TestCase):
    def test_reported_padded_alert_with_credential_request_no_longer_interrupts(self):
        unpadded = _routed(ALERT + " " + CREDENTIAL)
        self.assertEqual(_outcome(unpadded), VETO)

        padded = _routed(_pad(CAP) + ALERT + " " + CREDENTIAL)
        self.assertNotEqual(padded.result.action.value, "interrupt")
        self.assertFalse(padded.result.hard_veto_applied)
        self.assertTrue(padded.result.analysis_incomplete)
        self.assertEqual(_rule_ids(padded), [])  # not claimed as detected

    def test_credential_request_inside_the_analysed_part_vetoes_an_oversized_message(self):
        routed = _routed(CREDENTIAL + " " + _pad(CAP))
        self.assertEqual(_outcome(routed), VETO)
        self.assertIn("CRED_SECRET_DISCLOSURE", _rule_ids(routed))
        self.assertTrue(routed.result.analysis_incomplete)

    def test_credential_request_past_the_limit_is_not_claimed_as_detected(self):
        routed = _routed(_pad(CAP) + CREDENTIAL)
        self.assertEqual(_rule_ids(routed), [])
        self.assertFalse(routed.result.hard_veto_applied)
        self.assertTrue(routed.result.analysis_incomplete)
        self.assertNotEqual(routed.result.action.value, "interrupt")

    def test_credential_request_split_by_the_limit_is_not_claimed_as_detected(self):
        routed = _routed(_pad(CAP - 40) + CREDENTIAL + " " + _pad(500))
        self.assertEqual(_rule_ids(routed), [])
        self.assertNotEqual(routed.result.action.value, "interrupt")

    def test_severe_prefix_keeps_the_veto_over_an_urgent_looking_tail(self):
        routed = _routed(CREDENTIAL + " " + _pad(CAP) + ALERT)
        self.assertEqual(_outcome(routed), VETO)


class UrgencyPlacementTests(unittest.TestCase):
    def test_complete_urgent_message_interrupts(self):
        self.assertEqual(_outcome(_routed(ALERT)), INTERRUPT)

    def test_urgent_prefix_with_a_benign_unread_tail_interrupts(self):
        self.assertEqual(_outcome(_routed(ALERT + " " + _pad(CAP))), INTERRUPT)

    def test_moving_the_same_alert_past_the_limit_withdraws_the_interrupt(self):
        inside = _routed(ALERT + " " + _pad(CAP))
        outside = _routed(_pad(CAP) + ALERT)
        self.assertEqual(_outcome(inside), INTERRUPT)
        self.assertNotEqual(outside.result.action.value, "interrupt")
        # Same class and same urgency level: only eligibility differs.
        self.assertEqual(inside.classification.selected, outside.classification.selected)
        self.assertEqual(inside.features.level_of(FeatureFamily.LEGITIMATE_URGENCY),
                         outside.features.level_of(FeatureFamily.LEGITIMATE_URGENCY))

    def test_active_disruption_counts_only_inside_the_analysed_part(self):
        self.assertEqual(_outcome(_routed(DISRUPTION + " " + _pad(CAP))), INTERRUPT)
        self.assertNotEqual(_routed(_pad(CAP) + DISRUPTION).result.action.value, "interrupt")

    def test_near_term_event_cannot_complete_urgency_found_only_past_the_limit(self):
        inside = _routed_with_context(APPOINTMENT + " " + _pad(CAP), near_term_event=True)
        outside = _routed_with_context(_pad(CAP) + APPOINTMENT, near_term_event=True)
        self.assertEqual(_outcome(inside), INTERRUPT)
        self.assertNotEqual(outside.result.action.value, "interrupt")
        alone = _routed_with_context(_pad(CAP) + ALERT, near_term_event=True)
        self.assertNotEqual(alone.result.action.value, "interrupt")

    def test_direct_address_cannot_amplify_urgency_found_only_past_the_limit(self):
        inside = _routed_with_context(ALERT + " " + _pad(CAP), directly_addressed=True)
        outside = _routed_with_context(_pad(CAP) + ALERT, directly_addressed=True)
        self.assertEqual(_outcome(inside), INTERRUPT)
        self.assertNotEqual(outside.result.action.value, "interrupt")

    def test_structured_context_alone_is_unaffected_by_an_unread_tail(self):
        """Near-term plus direct address is structured, not content: coverage does not apply."""
        complete = _routed_with_context("See you there.", directly_addressed=True, near_term_event=True)
        padded = _routed_with_context(_pad(CAP) + ALERT, directly_addressed=True, near_term_event=True)
        self.assertEqual(_outcome(complete), INTERRUPT)
        self.assertEqual(_outcome(padded), INTERRUPT)
        self.assertFalse(padded.features.urgency_outside_safety_coverage)

    def test_relevance_never_comes_from_content(self):
        routed = _routed(_pad(CAP) + "Could you please reply to me, you specifically? " + ALERT)
        relevance = routed.features.get(FeatureFamily.RECIPIENT_RELEVANCE)
        self.assertTrue(relevance.contributors)
        for contributor in relevance.contributors:
            self.assertIs(contributor.source, SupportSource.STRUCTURED_METADATA)
        channel = _routed(_pad(CAP) + ALERT, surface="channel", participants=400)
        self.assertEqual(channel.features.level_of(FeatureFamily.RECIPIENT_RELEVANCE), FeatureLevel.LOW)
        self.assertNotEqual(channel.result.action.value, "interrupt")

    def test_a_silenced_thread_still_receives_an_analysed_alert_but_not_an_unread_one(self):
        silenced = {"silenced_conversations": ["conv-coverage"]}
        self.assertEqual(_outcome(_routed(ALERT + " " + _pad(CAP), preferences=silenced)), INTERRUPT)
        self.assertNotEqual(_routed(_pad(CAP) + ALERT, preferences=silenced).result.action.value, "interrupt")


class MixedContentTests(unittest.TestCase):
    def test_analysed_alert_with_an_unread_credential_tail_reports_its_gap_honestly(self):
        """The alert was analysed, so it may still interrupt; the tail is not claimed as checked."""
        routed = _routed(ALERT + " " + _pad(CAP) + CREDENTIAL)
        self.assertEqual(_rule_ids(routed), [])
        self.assertTrue(routed.result.analysis_incomplete)
        self.assertFalse(routed.features.urgency_outside_safety_coverage)

    def test_independent_severe_findings_in_two_sources_still_veto(self):
        routed = _routed(CREDENTIAL + " " + _pad(CAP), kind="audio", asr=CREDENTIAL + " " + _pad(CAP))
        self.assertEqual(_outcome(routed), VETO)
        provenances = {finding.provenance for finding in routed.assessment.findings}
        self.assertEqual(provenances, {Provenance.AUTHORED_TEXT, Provenance.ASR_TRANSCRIPT})

    def test_one_source_cannot_lend_its_coverage_to_another(self):
        authored_tail = _routed(_pad(CAP) + ALERT, kind="audio", asr="Club newsletter.")
        extracted_tail = _routed(_pad(CAP), kind="audio", asr=_pad(CAP) + ALERT)
        for routed in (authored_tail, extracted_tail):
            self.assertNotEqual(routed.result.action.value, "interrupt")
            self.assertTrue(routed.features.urgency_outside_safety_coverage)


class MediaProvenanceTests(unittest.TestCase):
    def test_transcript_alert_counts_only_inside_the_analysed_part(self):
        self.assertEqual(_outcome(_routed(kind="audio", asr=ALERT)), INTERRUPT)
        self.assertEqual(_outcome(_routed(kind="audio", asr=ALERT + " " + _pad(CAP))), INTERRUPT)
        self.assertNotEqual(_routed(kind="audio", asr=_pad(CAP) + ALERT).result.action.value, "interrupt")

    def test_image_text_alert_counts_only_inside_the_analysed_part(self):
        self.assertEqual(_outcome(_routed(kind="image", ocr=ALERT)), INTERRUPT)
        inside = _routed(kind="image", ocr=ALERT + " " + _pad(CAP), qr="https://example.invalid/a")
        outside = _routed(kind="image", ocr=_pad(CAP) + ALERT, qr="https://example.invalid/a")
        self.assertEqual(_outcome(inside), INTERRUPT)
        self.assertNotEqual(outside.result.action.value, "interrupt")

    def test_decoded_code_alert_past_the_limit_does_not_interrupt(self):
        routed = _routed(kind="image", ocr="Club newsletter.", qr=_pad(CAP) + ALERT)
        self.assertNotEqual(routed.result.action.value, "interrupt")
        self.assertTrue(routed.features.urgency_outside_safety_coverage)

    def test_unread_transcript_tail_credential_is_not_claimed(self):
        routed = _routed(kind="audio", asr=_pad(CAP) + CREDENTIAL)
        self.assertEqual(_rule_ids(routed), [])
        self.assertTrue(routed.result.analysis_incomplete)

    def test_fragments_from_unrelated_sources_are_not_joined(self):
        routed = _routed("Fire", kind="audio", asr="alarm")
        self.assertNotEqual(routed.classification.selected.value, "safety_alert")
        self.assertNotEqual(routed.result.action.value, "interrupt")


class GuardIsNarrowTests(unittest.TestCase):
    COMPLETE_MESSAGES = (
        ALERT, CREDENTIAL, DISRUPTION, APPOINTMENT, ALERT + " " + CREDENTIAL,
        "Spring sale: 30% off everything, shop now.", "Lunch on Friday? I booked a table by the river.",
    )

    def test_never_set_when_every_source_was_analysed_in_full(self):
        for body in self.COMPLETE_MESSAGES:
            routed = _routed(body)
            self.assertFalse(routed.features.urgency_outside_safety_coverage, body[:12])
            self.assertEqual(routed.classification.selected_within_safety_coverage,
                             routed.classification.selected)
            for contributor in routed.classification.contributors:
                self.assertFalse(contributor.outside_safety_coverage)

    def test_directly_built_features_default_to_covered(self):
        from message_router.analysis import DecisionFeatures

        self.assertFalse(DecisionFeatures().urgency_outside_safety_coverage)

    def test_rule_withholds_only_the_interrupt(self):
        """Flip only the coverage fact on a genuinely urgent message: nothing else moves."""
        routed = _routed(ALERT)
        self.assertEqual(_outcome(routed), INTERRUPT)
        flipped = dataclasses.replace(routed.features, urgency_outside_safety_coverage=True)
        result = route(routed.assessment, routed.classification, flipped)
        self.assertNotEqual(result.action.value, "interrupt")
        self.assertFalse(result.hard_veto_applied)

    def test_the_veto_is_reached_before_the_coverage_guard(self):
        routed = _routed(ALERT + " " + CREDENTIAL)
        flipped = dataclasses.replace(routed.features, urgency_outside_safety_coverage=True)
        self.assertEqual(route(routed.assessment, routed.classification, flipped).policy_id,
                         "route.safety.hard_veto")


class PrivacyDeterminismTests(unittest.TestCase):
    def test_repeated_routing_is_identical(self):
        body = _pad(CAP) + ALERT + " " + CREDENTIAL
        first = json.dumps(decision_to_dict(_routed(body)), sort_keys=True)
        for _ in range(3):
            self.assertEqual(json.dumps(decision_to_dict(_routed(body)), sort_keys=True), first)

    def test_unread_tail_never_reaches_the_decision(self):
        for body in (_pad(CAP) + ALERT + " " + MARKER, ALERT + " " + _pad(CAP) + MARKER):
            decision = json.dumps(decision_to_dict(_routed(body)), sort_keys=True)
            self.assertNotIn(MARKER, decision)
            self.assertNotIn("east wing", decision)

    def test_oversized_input_stays_bounded(self):
        started = time.perf_counter()
        routed = _routed(_pad(50 * CAP) + ALERT)
        self.assertLess(time.perf_counter() - started, 20.0)
        self.assertNotEqual(routed.result.action.value, "interrupt")


class RouteCommandTests(unittest.TestCase):
    """The reported case, through the public command line."""

    def _run(self, body: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "request.json"
            path.write_text(_document(body), encoding="utf-8")
            env = {**os.environ, "PYTHONPATH": str(SRC_DIR), "PYTHONDONTWRITEBYTECODE": "1"}
            return subprocess.run(
                [sys.executable, "-m", "message_router", "route", "--input", str(path)],
                capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=120,
            )

    def test_padded_alert_with_credential_request_is_not_an_interrupt(self):
        completed = self._run(_pad(CAP) + ALERT + " " + CREDENTIAL + " " + MARKER)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        decision = json.loads(completed.stdout)
        self.assertNotEqual(decision["action"], "interrupt")
        self.assertTrue(decision["analysis_incomplete"])
        self.assertEqual(decision["safety"]["findings"], [])
        self.assertNotIn(MARKER, completed.stdout)

    def test_the_same_request_unpadded_is_vetoed(self):
        completed = self._run(ALERT + " " + CREDENTIAL)
        decision = json.loads(completed.stdout)
        self.assertEqual((decision["action"], decision["policy_id"]), VETO)


if __name__ == "__main__":
    unittest.main()
