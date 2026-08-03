"""Behavioural tests for classification and decision features.

All scenario strings are written for this suite.
"""

from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from message_router.analysis import (  # noqa: E402
    ClassificationResult,
    Completeness,
    ContentAvailability,
    DecisionFeatures,
    FeatureFamily,
    FeatureLevel,
    FeatureNote,
    InteractionSummary,
    MessageClass,
    ModalityConsistency,
    SupportSource,
    classify,
    extract_features,
)
from message_router.domain import (  # noqa: E402
    ConversationContext,
    ConversationSurface,
    ExtractionStatus,
    IncomingMessage,
    MediaAnalysis,
    MediaKind,
    MediaReference,
    Provenance,
    QuietPeriod,
    RecipientPreferences,
    RecipientScope,
    SenderContext,
    TextContent,
    TrustCategory,
    TrustSignal,
    ValidationError,
)
from message_router.safety import Severity, analyze_message  # noqa: E402

WHEN = dt.datetime(2031, 3, 9, 14, 30, tzinfo=dt.timezone.utc)
READER = RecipientScope("rcp-4kq")
DIRECT = ConversationContext("conv-31b", ConversationSurface.DIRECT)
GROUP = ConversationContext("conv-88f", ConversationSurface.GROUP, 30)
IMAGE = MediaReference("att-55c", MediaKind.IMAGE)
AUDIO = MediaReference("att-77d", MediaKind.AUDIO)


def message(text=None, *, conversation=DIRECT, sender=None, attachment=None):
    return IncomingMessage(
        "note-7712", READER, conversation, WHEN,
        body=TextContent(text, Provenance.AUTHORED_TEXT) if text is not None else None,
        sender=sender, attachment=attachment,
    )


def pipeline(text=None, *, media=(), forwarding_count=None, **msg_kwargs):
    """Message -> safety -> classification -> features, as a caller would."""
    msg = message(text, **msg_kwargs)
    assessment = analyze_message(msg, media_analyses=media, forwarding_count=forwarding_count)
    texts = ((text, Provenance.AUTHORED_TEXT),) if text is not None else ()
    for analysis in media:
        if analysis.content is not None:
            texts = texts + ((analysis.content.value, analysis.content.provenance),)
    result = classify(texts, assessment, forwarding_count=forwarding_count)
    return msg, assessment, result


def classify_text(text, **kwargs):
    return classify(((text, Provenance.AUTHORED_TEXT),), **kwargs)


# ---------------------------------------------------------------------------
# Classification independence.
# ---------------------------------------------------------------------------


class ClassificationIndependenceTests(unittest.TestCase):
    def test_result_carries_no_route_or_final_outputs(self):
        result = classify_text("Could you please confirm?")
        for banned in ("action", "route", "confidence", "rationale", "reason", "score", "rank", "evidence"):
            self.assertFalse(hasattr(result, banned), banned)

    def test_classify_accepts_no_routing_action(self):
        import inspect

        self.assertNotIn("action", inspect.signature(classify).parameters)

    def test_same_class_can_pair_with_any_future_action(self):
        """Classification does not imply an action; the vocabularies are
        independent, so every pairing is representable."""
        from message_router.domain import RoutingAction

        result = classify_text("Could you please confirm?")
        for action in RoutingAction:
            self.assertIsNot(result.selected, action)
            self.assertNotIsInstance(action, MessageClass)

    def test_different_classes_can_share_a_future_action(self):
        first = classify_text("20% off, shop now")
        second = classify_text("Good morning!")
        self.assertIsNot(first.selected, second.selected)
        self.assertNotIsInstance(first.selected, type(None))

    def test_suspicious_classification_does_not_mutate_safety(self):
        msg, assessment, result = pipeline("Send me your OTP right now")
        before = (assessment.highest_severity, assessment.hard_veto_required, assessment.findings)
        self.assertIs(result.selected, MessageClass.SUSPICIOUS_CONTENT)
        self.assertEqual(
            (assessment.highest_severity, assessment.hard_veto_required, assessment.findings), before
        )
        self.assertIs(assessment.highest_severity, Severity.SEVERE)


# ---------------------------------------------------------------------------
# Classification contributors.
# ---------------------------------------------------------------------------


class ClassificationContributorTests(unittest.TestCase):
    CASES = (
        ("Could you please send the report?", MessageClass.DIRECT_REQUEST),
        ("Fire alarm activated, please evacuate the building", MessageClass.SAFETY_ALERT),
        ("Your appointment is scheduled for Tuesday at 10am", MessageClass.SCHEDULED_EVENT),
        ("Invoice 8841 - payment received", MessageClass.FINANCIAL_NOTICE),
        ("Your parcel is out for delivery", MessageClass.SERVICE_UPDATE),
        ("20% off our new collection, shop now", MessageClass.MARKETING),
        ("Good morning! Hope you're well", MessageClass.SOCIAL_MESSAGE),
        ("Forward this to everyone, don't break the chain", MessageClass.PROPAGATED_CONTENT),
    )

    def test_each_family_selects_its_class(self):
        for text, expected in self.CASES:
            self.assertIs(classify_text(text).selected, expected, text)

    def test_suspicious_content_from_safety_findings(self):
        _, _, result = pipeline("Send me your password immediately")
        self.assertIs(result.selected, MessageClass.SUSPICIOUS_CONTENT)

    def test_uncategorized_when_nothing_matches(self):
        result = classify_text("the blue kettle sits on the third shelf")
        self.assertIs(result.selected, MessageClass.UNCATEGORIZED)
        self.assertEqual(result.contributors, ())

    def test_no_content_reports_a_distinct_diagnostic(self):
        empty = classify(())
        self.assertIs(empty.selected, MessageClass.UNCATEGORIZED)
        self.assertEqual(empty.tie_break.value, "no_usable_content")
        nonmatching = classify_text("the blue kettle sits on the third shelf")
        self.assertEqual(nonmatching.tie_break.value, "no_contributor_reached_threshold")

    def test_ambiguity_recorded_when_contributors_disagree(self):
        result = classify_text("Your invoice is ready, could you please confirm payment?")
        self.assertTrue(result.ambiguous)
        self.assertGreater(len(result.considered), 1)

    def test_contributor_ordering_is_deterministic_and_sorted(self):
        text = "Your appointment is on Tuesday at 10am, could you please confirm?"
        first = classify_text(text)
        second = classify_text(text)
        self.assertEqual(first.contributors, second.contributors)
        keys = [c.sort_key() for c in first.contributors]
        self.assertEqual(keys, sorted(keys))

    def test_tie_break_is_stable_and_reported(self):
        text = "Your appointment is on Tuesday at 10am. Your invoice is ready."
        results = {classify_text(text).selected for _ in range(5)}
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(classify_text(text).tie_break)

    def test_provenance_is_preserved_per_contributor(self):
        result = classify(
            (("Could you please confirm?", Provenance.OCR_TEXT),),
        )
        self.assertTrue(result.contributors)
        self.assertTrue(all(c.provenance is Provenance.OCR_TEXT for c in result.contributors))

    def test_incomplete_analysis_is_preserved(self):
        _, assessment, result = pipeline(
            "Could you please confirm?",
            media=(MediaAnalysis(IMAGE, ExtractionStatus.FAILED),),
        )
        self.assertTrue(assessment.analysis_incomplete)
        self.assertTrue(result.analysis_incomplete)
        self.assertTrue(all(c.from_incomplete_analysis for c in result.contributors))

    def test_unknown_class_value_fails_closed(self):
        from message_router.domain import UnsupportedValueError

        with self.assertRaises(UnsupportedValueError):
            MessageClass.from_value("not_a_class")
        self.assertIs(MessageClass.from_value("marketing"), MessageClass.MARKETING)


# ---------------------------------------------------------------------------
# Urgency vs pressure.
# ---------------------------------------------------------------------------


class UrgencyTests(unittest.TestCase):
    def features_for(self, text, **kwargs):
        msg, assessment, result = pipeline(text, **kwargs)
        return extract_features(msg, assessment, result, **{
            k: v for k, v in kwargs.items() if k in ("preferences", "local_minute_of_day")
        })

    def test_imminent_safety_notice_is_high_urgency(self):
        f = self.features_for("Fire alarm activated, please evacuate the building")
        self.assertIs(f.level_of(FeatureFamily.LEGITIMATE_URGENCY), FeatureLevel.HIGH)

    def test_dated_commitment_carries_urgency(self):
        f = self.features_for("Your appointment is scheduled for Tuesday at 10am")
        self.assertIn(f.level_of(FeatureFamily.LEGITIMATE_URGENCY),
                      (FeatureLevel.MEDIUM, FeatureLevel.HIGH))

    def test_a_plain_request_carries_no_urgency(self):
        """Regression: request grammar is not a time-sensitivity signal.

        Being asked something is not being asked something urgently, and
        treating it as such hands out priority for phrasing alone.
        """
        for text in ("Could you please send the file?",
                     "Can you let me know what time works?",
                     "Please send the report when you can",
                     "I need you to check the figures"):
            f = self.features_for(text)
            self.assertIs(f.level_of(FeatureFamily.LEGITIMATE_URGENCY), FeatureLevel.ABSENT, text)

    def test_ordinary_content_is_not_urgent(self):
        for text in ("20% off our new collection, shop now", "Good morning! Hope you're well"):
            f = self.features_for(text)
            self.assertIs(f.level_of(FeatureFamily.LEGITIMATE_URGENCY), FeatureLevel.ABSENT, text)

    def test_shouting_and_punctuation_alone_create_no_urgency(self):
        for text in ("THE KETTLE IS ON THE SHELF", "the kettle is on the shelf!!!!!",
                     "urgent", "immediately"):
            f = self.features_for(text)
            self.assertIs(f.level_of(FeatureFamily.LEGITIMATE_URGENCY), FeatureLevel.ABSENT, text)

    def test_scam_urgency_becomes_pressure_not_urgency(self):
        f = self.features_for(
            "Your account will be suspended. Pay immediately via gift cards. Do not contact support."
        )
        self.assertIs(f.level_of(FeatureFamily.LEGITIMATE_URGENCY), FeatureLevel.ABSENT)
        self.assertIn(f.level_of(FeatureFamily.MALICIOUS_PRESSURE),
                      (FeatureLevel.MEDIUM, FeatureLevel.HIGH))

    def test_pressure_presence_is_recorded_as_withheld_urgency(self):
        f = self.features_for("Pay immediately or your account will be suspended")
        urgency = f.get(FeatureFamily.LEGITIMATE_URGENCY)
        notes = {c.note for c in urgency.contributors}
        self.assertIn(FeatureNote.URGENCY_WITHHELD_RISK_ATTRIBUTED, notes)

    def test_pressure_derives_only_from_safety_findings(self):
        f = self.features_for("Pay immediately or your account will be suspended")
        pressure = f.get(FeatureFamily.MALICIOUS_PRESSURE)
        self.assertTrue(pressure.contributors)
        for contributor in pressure.contributors:
            self.assertIs(contributor.source, SupportSource.SAFETY_FINDINGS)

    def test_legitimate_urgency_survives_quiet_hours_and_silenced_thread(self):
        prefs = RecipientPreferences(
            quiet_period=QuietPeriod(22 * 60, 7 * 60),
            muted_conversation_ids=frozenset({"conv-31b"}),
        )
        msg, assessment, result = pipeline("Fire alarm activated, please evacuate the building")
        f = extract_features(msg, assessment, result, preferences=prefs, local_minute_of_day=23 * 60)
        self.assertIs(f.level_of(FeatureFamily.LEGITIMATE_URGENCY), FeatureLevel.HIGH)
        self.assertIs(f.level_of(FeatureFamily.QUIET_HOURS_IMPACT), FeatureLevel.HIGH)
        self.assertIs(f.level_of(FeatureFamily.SILENCED_CONVERSATION), FeatureLevel.HIGH)


class UrgencyOriginationTests(unittest.TestCase):
    """What is allowed to *create* legitimate urgency, and what only amplifies it.

    Regression: a plain request classified as ``DIRECT_REQUEST`` originated
    urgency, and direct targeting then amplified it to ``high``, so an
    ordinary "could you send this sometime" reached the interrupt threshold
    on phrasing plus metadata alone.
    """

    def features_for(self, text, *, directly_addressed=False, near_term_event=False):
        msg = message(text)
        assessment = analyze_message(msg)
        result = classify(((text, Provenance.AUTHORED_TEXT),), assessment,
                          directly_addressed=directly_addressed)
        return extract_features(msg, assessment, result,
                                directly_addressed=directly_addressed,
                                near_term_event=near_term_event)

    def urgency_rules(self, f):
        return {c.rule_id for c in f.get(FeatureFamily.LEGITIMATE_URGENCY).contributors}

    def test_a_request_alone_originates_no_urgency(self):
        f = self.features_for("Could you please send the report sometime?")
        self.assertIs(f.level_of(FeatureFamily.LEGITIMATE_URGENCY), FeatureLevel.ABSENT)
        self.assertEqual(self.urgency_rules(f), set())

    def test_targeting_alone_originates_no_urgency(self):
        for text in ("Could you please send the report sometime?",
                     "20% off our new collection, shop now",
                     "the kettle is on the shelf"):
            f = self.features_for(text, directly_addressed=True)
            self.assertIs(f.level_of(FeatureFamily.LEGITIMATE_URGENCY), FeatureLevel.ABSENT, text)
            self.assertNotIn("URGENCY_DIRECT_TARGET", self.urgency_rules(f), text)

    def test_targeting_still_contributes_relevance(self):
        plain = self.features_for("Could you please send the report sometime?")
        targeted = self.features_for("Could you please send the report sometime?",
                                     directly_addressed=True)
        self.assertIs(plain.level_of(FeatureFamily.RECIPIENT_RELEVANCE), FeatureLevel.MEDIUM)
        self.assertIs(targeted.level_of(FeatureFamily.RECIPIENT_RELEVANCE), FeatureLevel.HIGH)

    def test_targeting_amplifies_only_independently_established_urgency(self):
        independent = self.features_for("Could you please send the report?", near_term_event=True)
        amplified = self.features_for("Could you please send the report?",
                                      near_term_event=True, directly_addressed=True)
        self.assertIs(independent.level_of(FeatureFamily.LEGITIMATE_URGENCY), FeatureLevel.MEDIUM)
        self.assertIs(amplified.level_of(FeatureFamily.LEGITIMATE_URGENCY), FeatureLevel.HIGH)
        self.assertIn("URGENCY_DIRECT_TARGET", self.urgency_rules(amplified))

    def test_targeting_never_moves_absent_or_unknown_upward(self):
        """Targeting must not turn absent into low, or unknown into
        established. With nothing to amplify it adds no contributor at all."""
        for text in ("the kettle is on the shelf", "THE KETTLE IS ON THE SHELF",
                     "the kettle is on the shelf!!!!!", "urgent", "immediately"):
            plain = self.features_for(text)
            targeted = self.features_for(text, directly_addressed=True)
            self.assertIs(plain.level_of(FeatureFamily.LEGITIMATE_URGENCY), FeatureLevel.ABSENT, text)
            self.assertIs(targeted.level_of(FeatureFamily.LEGITIMATE_URGENCY),
                          FeatureLevel.ABSENT, text)


class ActiveDisruptionTests(unittest.TestCase):
    """A service the reader depends on being down *now* is urgency-bearing.

    Everything adjacent to it deliberately is not: a maintenance notice, a
    resolved incident, a retrospective, a rumour, and an ordinary status
    update all describe something that is not happening to the reader right
    now.
    """

    ACTIVE = (
        "Service alert: the platform is currently down and engineers are responding.",
        "We are experiencing a major outage affecting logins right now.",
        "The payment system is currently unavailable; there is an ongoing incident.",
    )
    NOT_ACTIVE = (
        "Scheduled maintenance window next week may cause downtime.",
        "The outage has been resolved and service is restored.",
        "The platform was down yesterday for about an hour.",
        "I heard that the platform might be down, not sure.",
        "Your parcel is out for delivery today.",
        "Your account has been updated successfully.",
        "This guide explains what to do when the service is down.",
    )

    def features_for(self, text):
        msg = message(text)
        assessment = analyze_message(msg)
        result = classify(((text, Provenance.AUTHORED_TEXT),), assessment)
        return result, extract_features(msg, assessment, result)

    def test_active_disruption_originates_urgency(self):
        for text in self.ACTIVE:
            result, f = self.features_for(text)
            self.assertIn("CLS_ACTIVE_DISRUPTION", {c.rule_id for c in result.contributors}, text)
            self.assertIs(f.level_of(FeatureFamily.LEGITIMATE_URGENCY), FeatureLevel.HIGH, text)

    def test_everything_adjacent_to_it_does_not(self):
        for text in self.NOT_ACTIVE:
            result, f = self.features_for(text)
            self.assertNotIn("CLS_ACTIVE_DISRUPTION", {c.rule_id for c in result.contributors}, text)
            self.assertIs(f.level_of(FeatureFamily.LEGITIMATE_URGENCY), FeatureLevel.ABSENT, text)

    def test_disruption_urgency_keeps_its_provenance(self):
        text = "Service alert: the platform is currently down and engineers are responding."
        msg = message(attachment=IMAGE)
        analysis = MediaAnalysis(IMAGE, ExtractionStatus.SUCCEEDED,
                                 content=TextContent(text, Provenance.OCR_TEXT))
        assessment = analyze_message(msg, media_analyses=(analysis,))
        result = classify(((text, Provenance.OCR_TEXT),), assessment)
        f = extract_features(msg, assessment, result)
        urgency = f.get(FeatureFamily.LEGITIMATE_URGENCY)
        disruption = [c for c in urgency.contributors if c.rule_id == "URGENCY_ACTIVE_DISRUPTION"]
        self.assertTrue(disruption)
        self.assertIs(disruption[0].provenance, Provenance.OCR_TEXT)

    def test_scam_pressure_stays_separate_from_disruption_urgency(self):
        text = ("The service is currently down. Pay immediately via gift cards to restore access. "
                "Do not contact support.")
        msg = message(text)
        assessment = analyze_message(msg)
        result = classify(((text, Provenance.AUTHORED_TEXT),), assessment)
        f = extract_features(msg, assessment, result)
        self.assertIn(f.level_of(FeatureFamily.MALICIOUS_PRESSURE),
                      (FeatureLevel.MEDIUM, FeatureLevel.HIGH))
        pressure = f.get(FeatureFamily.MALICIOUS_PRESSURE)
        for contributor in pressure.contributors:
            self.assertIs(contributor.source, SupportSource.SAFETY_FINDINGS)
        urgency = f.get(FeatureFamily.LEGITIMATE_URGENCY)
        self.assertIn(FeatureNote.URGENCY_WITHHELD_RISK_ATTRIBUTED,
                      {c.note for c in urgency.contributors})


class RiskFamilyProjectionTests(unittest.TestCase):
    """Independent risk families, projected from findings and deduplicated."""

    def families_for(self, text):
        msg = message(text)
        assessment = analyze_message(msg)
        result = classify(((text, Provenance.AUTHORED_TEXT),), assessment)
        return extract_features(msg, assessment, result).risk_families

    def test_ordinary_content_projects_no_families(self):
        for text in ("Invoice number 8841 for your subscription. Due date is Friday.",
                     "Your parcel is out for delivery today",
                     "Good morning! Hope you're well"):
            self.assertEqual(self.families_for(text), frozenset(), text)

    def test_independent_indicators_project_distinct_families(self):
        from message_router.analysis import RiskFamily

        families = self.families_for("Pay now using gift cards to avoid a penalty.")
        self.assertIn(RiskFamily.COERCIVE_PAYMENT, families)
        self.assertIn(RiskFamily.UNUSUAL_PAYMENT_CHANNEL, families)

    def test_weak_and_ambiguous_findings_never_qualify(self):
        for text in ("Have a look at this",
                     "I forgot my password again this morning",
                     "The report is due on Friday, please send it next week"):
            self.assertEqual(self.families_for(text), frozenset(), text)

    def test_repeated_expressions_of_one_indicator_count_once(self):
        from message_router.analysis import RiskFamily

        # Two threat phrasings, one underlying pressure indicator.
        families = self.families_for(
            "Your account will be suspended and legal action will follow unless you pay now."
        )
        self.assertIn(RiskFamily.MALICIOUS_PRESSURE, families)
        self.assertEqual(
            len([f for f in families if f is RiskFamily.MALICIOUS_PRESSURE]), 1
        )

    def test_root_cause_key_is_structured_and_content_free(self):
        from message_router.analysis import RiskRootCause
        import dataclasses

        names = {f.name for f in dataclasses.fields(RiskRootCause)}
        self.assertEqual(names, {"family", "rule_id", "diagnostic"})
        for term in ("text", "content", "payload", "url", "value", "provenance", "source"):
            for name in names:
                self.assertNotIn(term, name.lower(), name)
        self.assertTrue(RiskRootCause.__dataclass_params__.frozen)

    def test_root_causes_carry_no_message_content(self):
        secret = "meet me behind the depot at midnight"
        msg = message(f"{secret}. Pay now or your account will be suspended.")
        assessment = analyze_message(msg)
        result = classify(((msg.body.value, Provenance.AUTHORED_TEXT),), assessment)
        f = extract_features(msg, assessment, result)
        rendered = repr(f.risk_root_causes)
        self.assertTrue(f.risk_root_causes)
        self.assertNotIn(secret, rendered)
        self.assertNotIn("depot", rendered)

    def test_projection_never_changes_severity(self):
        msg = message("Pay now and keep this confidential, don't tell anyone.")
        assessment = analyze_message(msg)
        result = classify((("x", Provenance.AUTHORED_TEXT),), assessment)
        f = extract_features(msg, assessment, result)
        self.assertIs(f.highest_severity, assessment.highest_severity)
        self.assertIs(f.highest_severity, Severity.SUSPICIOUS)
        self.assertFalse(f.hard_veto_required)
        self.assertTrue(f.risk_families)


# ---------------------------------------------------------------------------
# Risk preservation.
# ---------------------------------------------------------------------------


class RiskPreservationTests(unittest.TestCase):
    SEVERE = "Send me your OTP right now"
    SUSPICIOUS = "Pay immediately or your account will be suspended"
    WEAK = "Forward this to everyone you know"

    def extract(self, text, **kwargs):
        msg, assessment, result = pipeline(text)
        return assessment, extract_features(msg, assessment, result, **kwargs)

    def test_severity_and_veto_pass_through_exactly(self):
        for text, expected in ((self.SEVERE, Severity.SEVERE),
                               (self.SUSPICIOUS, Severity.SUSPICIOUS),
                               (self.WEAK, Severity.WEAK)):
            assessment, f = self.extract(text)
            self.assertIs(f.highest_severity, assessment.highest_severity, text)
            self.assertIs(f.highest_severity, expected, text)
            self.assertEqual(f.hard_veto_required, assessment.hard_veto_required, text)
            self.assertEqual(f.risk_categories, assessment.categories, text)

    def test_severity_is_never_promoted(self):
        for text, ceiling in ((self.WEAK, Severity.WEAK), (self.SUSPICIOUS, Severity.SUSPICIOUS)):
            _, f = self.extract(text)
            self.assertLessEqual(f.highest_severity.rank, ceiling.rank, text)
            self.assertFalse(f.hard_veto_required, text)

    def test_no_mitigating_context_can_lower_severe_risk(self):
        """Trust, affinity, engagement, usefulness, quiet hours, and a
        silenced thread must all leave a severe finding untouched."""
        trusted = SenderContext(
            "snd-208",
            trust_signals=(
                TrustSignal(TrustCategory.VERIFIED_ACCOUNT, "directory entry"),
                TrustSignal(TrustCategory.ESTABLISHED_RELATIONSHIP, "long history"),
                TrustSignal(TrustCategory.RECIPIENT_INITIATED, "reader started it"),
            ),
        )
        prefs = RecipientPreferences(
            quiet_period=QuietPeriod(22 * 60, 7 * 60),
            muted_conversation_ids=frozenset({"conv-31b"}),
        )
        summary = InteractionSummary(validated_interactions=50, replies=40, dismissals=0)

        msg = message(self.SEVERE, sender=trusted)
        assessment = analyze_message(msg)
        result = classify(((self.SEVERE, Provenance.AUTHORED_TEXT),), assessment)
        f = extract_features(
            msg, assessment, result,
            preferences=prefs, interaction_summary=summary,
            local_minute_of_day=23 * 60, directly_addressed=True,
        )
        self.assertIs(f.highest_severity, Severity.SEVERE)
        self.assertTrue(f.hard_veto_required)
        self.assertIs(f.level_of(FeatureFamily.SAFETY_RISK), FeatureLevel.HIGH)
        # And the mitigating context is still reported, just not applied.
        self.assertIs(f.level_of(FeatureFamily.VALIDATED_TRUST), FeatureLevel.HIGH)
        self.assertIs(f.level_of(FeatureFamily.RECIPIENT_AFFINITY), FeatureLevel.HIGH)

    def test_extraction_does_not_mutate_the_assessment(self):
        msg, assessment, result = pipeline(self.SEVERE)
        snapshot = (
            assessment.findings, assessment.modality_outcomes, assessment.modality_conflicts,
            assessment.diagnostics, assessment.analysis_incomplete,
        )
        extract_features(msg, assessment, result)
        self.assertEqual(
            (assessment.findings, assessment.modality_outcomes, assessment.modality_conflicts,
             assessment.diagnostics, assessment.analysis_incomplete),
            snapshot,
        )

    def test_incomplete_safety_analysis_is_visible(self):
        msg, assessment, result = pipeline(
            "hello there", media=(MediaAnalysis(IMAGE, ExtractionStatus.FAILED),)
        )
        f = extract_features(msg, assessment, result)
        self.assertIs(f.level_of(FeatureFamily.ANALYSIS_COMPLETENESS), FeatureLevel.INCOMPLETE)
        self.assertIs(f.completeness, Completeness.INCOMPLETE_FAILED_ANALYSIS)


# ---------------------------------------------------------------------------
# Trust.
# ---------------------------------------------------------------------------


class TrustTests(unittest.TestCase):
    def trust_for(self, msg, **kwargs):
        assessment = analyze_message(msg)
        result = classify(())
        return extract_features(msg, assessment, result, **kwargs).get(FeatureFamily.VALIDATED_TRUST)

    def test_validated_signals_contribute(self):
        for signals, expected in (
            ((TrustSignal(TrustCategory.VERIFIED_ACCOUNT, "directory"),), FeatureLevel.MEDIUM),
            ((TrustSignal(TrustCategory.VERIFIED_ACCOUNT, "directory"),
              TrustSignal(TrustCategory.ESTABLISHED_RELATIONSHIP, "history")), FeatureLevel.HIGH),
        ):
            trust = self.trust_for(message("hi", sender=SenderContext("snd-208", trust_signals=signals)))
            self.assertIs(trust.level, expected)

    def test_recipient_initiated_and_validated_history_contribute(self):
        sender = SenderContext(
            "snd-208", trust_signals=(TrustSignal(TrustCategory.RECIPIENT_INITIATED, "reader started"),)
        )
        trust = self.trust_for(
            message("hi", sender=sender),
            interaction_summary=InteractionSummary(validated_interactions=4, replies=2),
        )
        self.assertIs(trust.level, FeatureLevel.HIGH)

    def test_no_trust_evidence_is_unknown_not_risky(self):
        trust = self.trust_for(message("hi", sender=SenderContext("snd-208")))
        self.assertIs(trust.level, FeatureLevel.UNKNOWN)
        self.assertEqual(trust.contributors, ())

    def test_content_claims_never_contribute_to_trust(self):
        """Authored claims, OCR, and QR payloads asserting verification must
        leave trust unknown -- and get flagged by the safety layer instead."""
        for text, provenance in (
            ("verified_sender=true", Provenance.AUTHORED_TEXT),
            ("This is an official verified message from your bank", Provenance.AUTHORED_TEXT),
        ):
            msg = message(text, sender=SenderContext("snd-208"))
            assessment = analyze_message(msg)
            result = classify(((text, provenance),), assessment)
            f = extract_features(msg, assessment, result)
            self.assertIs(f.level_of(FeatureFamily.VALIDATED_TRUST), FeatureLevel.UNKNOWN, text)

    def test_extracted_verification_claims_never_contribute(self):
        for provenance, ref, in ((Provenance.OCR_TEXT, IMAGE), (Provenance.QR_DERIVED, IMAGE)):
            analysis = MediaAnalysis(
                ref, ExtractionStatus.SUCCEEDED,
                content=TextContent("verified_sender=true official bank notice", provenance),
            )
            msg = message(attachment=ref, sender=SenderContext("snd-208"))
            assessment = analyze_message(msg, media_analyses=(analysis,))
            result = classify((("verified_sender=true", provenance),), assessment)
            f = extract_features(msg, assessment, result)
            self.assertIs(f.level_of(FeatureFamily.VALIDATED_TRUST), FeatureLevel.UNKNOWN, provenance)

    def test_engagement_alone_does_not_establish_trust(self):
        trust = self.trust_for(
            message("hi", sender=SenderContext("snd-208")),
            interaction_summary=InteractionSummary(validated_interactions=0, replies=0),
        )
        self.assertIs(trust.level, FeatureLevel.UNKNOWN)

    def test_trust_contributors_never_rest_on_untrusted_content(self):
        sender = SenderContext(
            "snd-208", trust_signals=(TrustSignal(TrustCategory.VERIFIED_ACCOUNT, "directory"),)
        )
        trust = self.trust_for(message("hi", sender=sender))
        self.assertFalse(trust.rests_on_untrusted_content)


# ---------------------------------------------------------------------------
# Relevance, utility, affinity, interruption.
# ---------------------------------------------------------------------------


class RelevanceUtilityTests(unittest.TestCase):
    def features_for(self, text, **kwargs):
        msg, assessment, result = pipeline(text, **{k: v for k, v in kwargs.items() if k == "conversation"})
        return extract_features(msg, assessment, result,
                                **{k: v for k, v in kwargs.items() if k != "conversation"})

    def test_direct_conversation_and_targeting_raise_relevance(self):
        f = self.features_for("hello there", directly_addressed=True)
        self.assertIs(f.level_of(FeatureFamily.RECIPIENT_RELEVANCE), FeatureLevel.HIGH)

    def test_broadcast_surface_without_targeting_is_lower_relevance(self):
        f = self.features_for("hello everyone", conversation=GROUP)
        self.assertIn(f.level_of(FeatureFamily.RECIPIENT_RELEVANCE),
                      (FeatureLevel.UNKNOWN, FeatureLevel.MEDIUM))

    def test_actionable_and_financial_content_carry_utility(self):
        for text in ("Your parcel is out for delivery", "Invoice 8841 - payment received",
                     "Your appointment is scheduled for Tuesday at 10am"):
            f = self.features_for(text)
            self.assertIs(f.level_of(FeatureFamily.PRACTICAL_UTILITY), FeatureLevel.MEDIUM, text)

    def test_marketing_is_low_utility_but_may_be_relevant(self):
        f = self.features_for("20% off our new collection, shop now", directly_addressed=True)
        self.assertIs(f.level_of(FeatureFamily.PRACTICAL_UTILITY), FeatureLevel.LOW)
        self.assertIs(f.level_of(FeatureFamily.RECIPIENT_RELEVANCE), FeatureLevel.HIGH)
        self.assertIs(f.level_of(FeatureFamily.LEGITIMATE_URGENCY), FeatureLevel.ABSENT)

    def test_suspicious_but_useful_preserves_both_independently(self):
        text = "Your parcel is out for delivery. Pay immediately or it will be returned."
        msg, assessment, result = pipeline(text)
        f = extract_features(msg, assessment, result)
        self.assertNotEqual(f.level_of(FeatureFamily.PRACTICAL_UTILITY), FeatureLevel.ABSENT)
        self.assertNotEqual(f.level_of(FeatureFamily.SAFETY_RISK), FeatureLevel.ABSENT)

    def test_absent_context_is_unknown(self):
        f = self.features_for("hello there")
        self.assertIs(f.level_of(FeatureFamily.RECIPIENT_AFFINITY), FeatureLevel.UNKNOWN)


class AffinityAndInterruptionTests(unittest.TestCase):
    def test_validated_summary_drives_affinity(self):
        msg, assessment, result = pipeline("hello")
        high = extract_features(msg, assessment, result,
                                interaction_summary=InteractionSummary(10, replies=8, dismissals=1))
        low = extract_features(msg, assessment, result,
                               interaction_summary=InteractionSummary(10, replies=0, dismissals=9))
        self.assertIs(high.level_of(FeatureFamily.RECIPIENT_AFFINITY), FeatureLevel.HIGH)
        self.assertIs(low.level_of(FeatureFamily.RECIPIENT_AFFINITY), FeatureLevel.LOW)

    def test_missing_summary_is_unknown_and_noted(self):
        msg, assessment, result = pipeline("hello")
        f = extract_features(msg, assessment, result)
        affinity = f.get(FeatureFamily.RECIPIENT_AFFINITY)
        self.assertIs(affinity.level, FeatureLevel.UNKNOWN)
        self.assertIn(FeatureNote.NO_HISTORY_SUMMARY, {c.note for c in affinity.contributors})

    def test_interruption_cost_exists_without_any_route_decision(self):
        prefs = RecipientPreferences(quiet_period=QuietPeriod(22 * 60, 7 * 60))
        msg, assessment, result = pipeline("Good morning! Hope you're well")
        f = extract_features(msg, assessment, result, preferences=prefs, local_minute_of_day=23 * 60)
        # A perfectly safe, non-urgent message can still be expensive to deliver now.
        self.assertIs(f.level_of(FeatureFamily.SAFETY_RISK), FeatureLevel.ABSENT)
        self.assertIs(f.level_of(FeatureFamily.INTERRUPTION_COST), FeatureLevel.MEDIUM)
        self.assertFalse(hasattr(f, "action"))

    def test_quiet_hours_and_silenced_state_are_separate_families(self):
        prefs = RecipientPreferences(
            quiet_period=QuietPeriod(22 * 60, 7 * 60), muted_conversation_ids=frozenset({"conv-31b"})
        )
        msg, assessment, result = pipeline("hello")
        f = extract_features(msg, assessment, result, preferences=prefs, local_minute_of_day=9 * 60)
        self.assertIs(f.level_of(FeatureFamily.QUIET_HOURS_IMPACT), FeatureLevel.ABSENT)
        self.assertIs(f.level_of(FeatureFamily.SILENCED_CONVERSATION), FeatureLevel.HIGH)

    def test_quiet_hours_unknown_without_supplied_local_time(self):
        prefs = RecipientPreferences(quiet_period=QuietPeriod(22 * 60, 7 * 60))
        msg, assessment, result = pipeline("hello")
        f = extract_features(msg, assessment, result, preferences=prefs)
        quiet = f.get(FeatureFamily.QUIET_HOURS_IMPACT)
        self.assertIs(quiet.level, FeatureLevel.UNKNOWN)
        self.assertIn(FeatureNote.QUIET_HOURS_UNKNOWN, {c.note for c in quiet.contributors})


# ---------------------------------------------------------------------------
# Repetition.
# ---------------------------------------------------------------------------


class RepetitionTests(unittest.TestCase):
    def test_forwarding_metadata_tiers(self):
        for count, expected in ((0, FeatureLevel.ABSENT), (2, FeatureLevel.LOW), (9, FeatureLevel.HIGH)):
            msg, assessment, result = pipeline("hello", forwarding_count=count)
            f = extract_features(msg, assessment, result, forwarding_count=count)
            self.assertIs(f.level_of(FeatureFamily.REPETITION), expected, count)

    def test_forwarding_alone_is_never_severe(self):
        msg, assessment, result = pipeline("Forward this to everyone", forwarding_count=50)
        f = extract_features(msg, assessment, result, forwarding_count=50)
        self.assertIsNot(f.highest_severity, Severity.SEVERE)
        self.assertFalse(f.hard_veto_required)

    def test_unknown_when_no_metadata_supplied(self):
        msg, assessment, result = pipeline("hello")
        f = extract_features(msg, assessment, result)
        self.assertIs(f.level_of(FeatureFamily.REPETITION), FeatureLevel.UNKNOWN)


# ---------------------------------------------------------------------------
# Modalities, completeness, availability.
# ---------------------------------------------------------------------------


class ModalityTests(unittest.TestCase):
    def ocr(self, text, status=ExtractionStatus.SUCCEEDED):
        content = TextContent(text, Provenance.OCR_TEXT) if text is not None else None
        return MediaAnalysis(IMAGE, status, content=content)

    def features(self, text=None, media=(), attachment=None):
        msg = message(text, attachment=attachment)
        assessment = analyze_message(msg, media_analyses=media)
        return extract_features(msg, assessment, classify(()))

    def test_text_only_and_media_only_and_mixed(self):
        text_only = self.features("hello there friend today")
        self.assertIs(text_only.content_availability, ContentAvailability.AUTHORED_ONLY)

        media_only = self.features(attachment=IMAGE, media=(self.ocr("a printed notice board"),))
        self.assertIs(media_only.content_availability, ContentAvailability.EXTRACTED_ONLY)

        mixed = self.features("look at this notice board printed", attachment=IMAGE,
                              media=(self.ocr("a printed notice board"),))
        self.assertIs(mixed.content_availability, ContentAvailability.AUTHORED_AND_EXTRACTED)

    def test_consistent_sources(self):
        f = self.features("printed notice board about parking", attachment=IMAGE,
                          media=(self.ocr("printed notice board about parking rules"),))
        self.assertIs(f.modality_consistency, ModalityConsistency.CONSISTENT)

    def test_conflicting_sources_are_reported_and_raise_ambiguity(self):
        f = self.features("dinner plans tomorrow evening around seven", attachment=IMAGE,
                          media=(self.ocr("quarterly warehouse inventory reconciliation spreadsheet"),))
        self.assertIs(f.modality_consistency, ModalityConsistency.CONFLICTING)
        self.assertIs(f.content_availability, ContentAvailability.CONFLICTED_CONTENT)
        self.assertIs(f.level_of(FeatureFamily.AMBIGUITY), FeatureLevel.HIGH)
        self.assertIs(f.completeness, Completeness.CONFLICTED)

    def test_degraded_states_map_to_distinct_outcomes(self):
        cases = (
            (ExtractionStatus.BACKEND_UNAVAILABLE, ModalityConsistency.UNAVAILABLE,
             Completeness.INCOMPLETE_UNAVAILABLE_BACKEND),
            (ExtractionStatus.FAILED, ModalityConsistency.FAILED, Completeness.INCOMPLETE_FAILED_ANALYSIS),
        )
        for status, consistency, completeness in cases:
            f = self.features("hello there", attachment=IMAGE, media=(self.ocr(None, status),))
            self.assertIs(f.modality_consistency, consistency, status)
            self.assertIs(f.completeness, completeness, status)

    def test_empty_extraction_is_not_evidence_of_safety(self):
        f = self.features("hello there", attachment=IMAGE,
                          media=(self.ocr(None, ExtractionStatus.NO_CONTENT_FOUND),))
        self.assertIsNot(f.level_of(FeatureFamily.MODALITY_CONSISTENCY), FeatureLevel.HIGH)

    def test_severe_finding_survives_another_modality_failing(self):
        strong = MediaAnalysis(
            IMAGE, ExtractionStatus.SUCCEEDED,
            content=TextContent("send me your password now", Provenance.OCR_TEXT),
        )
        failed = MediaAnalysis(AUDIO, ExtractionStatus.FAILED)
        msg = message(attachment=IMAGE)
        assessment = analyze_message(msg, media_analyses=(strong, failed))
        f = extract_features(msg, assessment, classify(()))
        self.assertIs(f.highest_severity, Severity.SEVERE)
        self.assertTrue(f.hard_veto_required)
        self.assertIs(f.completeness, Completeness.INCOMPLETE_FAILED_ANALYSIS)

    def test_conflict_representation_is_deterministic(self):
        args = ("dinner plans tomorrow evening around seven", IMAGE,
                (self.ocr("quarterly warehouse inventory reconciliation spreadsheet"),))
        first = self.features(args[0], attachment=args[1], media=args[2])
        second = self.features(args[0], attachment=args[1], media=args[2])
        self.assertEqual(first, second)


# ---------------------------------------------------------------------------
# Robustness.
# ---------------------------------------------------------------------------


class RobustnessTests(unittest.TestCase):
    def test_absent_authored_body_is_supported(self):
        msg = message(attachment=IMAGE)
        assessment = analyze_message(msg)
        f = extract_features(msg, assessment, classify(()))
        self.assertIsNone(msg.body)
        self.assertIs(f.content_availability, ContentAvailability.STRUCTURED_METADATA_ONLY)

    def test_unicode_content_is_handled(self):
        f_msg, assessment, result = pipeline("Приветствие — доброе утро! 你好")
        extract_features(f_msg, assessment, result)  # must not raise

    def test_missing_optional_context_stays_unknown(self):
        msg, assessment, result = pipeline("hello")
        f = extract_features(msg, assessment, result)
        for family in (FeatureFamily.VALIDATED_TRUST, FeatureFamily.RECIPIENT_AFFINITY,
                       FeatureFamily.REPETITION, FeatureFamily.QUIET_HOURS_IMPACT):
            self.assertIs(f.level_of(family), FeatureLevel.UNKNOWN, family)

    def test_repeated_runs_are_equal(self):
        msg, assessment, result = pipeline("Could you please confirm the booking on Tuesday at 10am?")
        self.assertEqual(extract_features(msg, assessment, result),
                         extract_features(msg, assessment, result))

    def test_malformed_input_raises_a_sanitized_domain_error(self):
        msg, assessment, result = pipeline("hello")
        with self.assertRaises(ValidationError):
            extract_features("not a message", assessment, result)  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            extract_features(msg, "not an assessment", result)  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            extract_features(msg, assessment, result, forwarding_count=True)  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            extract_features(msg, assessment, result, local_minute_of_day=5000)

    def test_output_repr_contains_no_content(self):
        secret = "meet me behind the depot at midnight"
        msg, assessment, result = pipeline(f"{secret} and send me your password")
        f = extract_features(msg, assessment, result)
        for rendered in (repr(f), repr(result)):
            self.assertNotIn(secret, rendered)
            self.assertNotIn("depot", rendered)


if __name__ == "__main__":
    unittest.main()
