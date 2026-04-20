"""
Pure-Python smoke test for the conflict typer.

No BERT, no torch — exercises the rule router on hand-crafted
StancedPair + Claim pairs so every branch (none / hard_contradiction /
temporal / opinion / ambiguity / misinformation / noise) is covered.

Run:
    python scripts/smoke_test_conflict_typing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.p2 import Claim, StancedPair, type_pair  # noqa: E402


def _sp(signal="conflict", stance="oppose", nli="contradiction", score=0.8):
    return StancedPair(
        claim_a_id="a",
        claim_b_id="b",
        nli_label=nli,
        stance_label=stance,
        stance_decision_score=score,
        stance_direction="a_as_claim",
        is_filtered=False,
        agreement_signal=signal,
        fusion_confidence=score,
    )


def _claim(cid="a", text="", **kw) -> Claim:
    return Claim(claim_id=cid, text=text, **kw)


def test_none_when_agreement() -> None:
    tp = type_pair(_sp(signal="agreement", stance="support", nli="entailment"),
                   _claim(), _claim())
    assert tp.conflict_type == "none"
    assert tp.resolution_policy == "pass_through"


def test_noise_when_filtered() -> None:
    tp = type_pair(_sp(signal="unrelated", stance="filtered", nli=None),
                   _claim(), _claim())
    assert tp.conflict_type == "noise"
    assert tp.resolution_policy == "skip"


def test_temporal_wins_when_years_differ() -> None:
    a = _claim(cid="a", text="The deal happened in 2014.", time="2014")
    b = _claim(cid="b", text="The deal happened in 2013.", time="2013")
    tp = type_pair(_sp(), a, b)
    assert tp.conflict_type == "temporal_conflict"
    assert tp.resolution_policy == "prefer_latest"


def test_ambiguity_when_subjects_differ() -> None:
    a = _claim(cid="a", text="Irrelevant", subject="Michael Jordan")
    b = _claim(cid="b", text="Irrelevant", subject="Michael I. Jordan")
    tp = type_pair(_sp(), a, b)
    assert tp.conflict_type == "ambiguity"
    assert tp.resolution_policy == "disambiguate_first"


def test_ambiguity_from_same_name_disambiguation() -> None:
    a = _claim(cid="a", text="Michael Jordan won the case in court.")
    b = _claim(cid="b", text="Michael I. Jordan denied the accusation in court.")
    tp = type_pair(_sp(), a, b)
    assert tp.conflict_type == "ambiguity"
    assert any("subjects differ" in r for r in tp.rationale)


def test_misinformation_from_medium_gap() -> None:
    a = _claim(cid="a", text="Claim here",
               source_metadata={"source_medium": "blog"})
    b = _claim(cid="b", text="Claim here",
               source_metadata={"source_medium": "official"})
    tp = type_pair(_sp(), a, b)
    assert tp.conflict_type == "misinformation"
    assert tp.resolution_policy == "down_weight_low_quality"


def test_misinformation_with_role_prior_fallback() -> None:
    # Query side has no source_medium; role prior should still allow a
    # large authority gap against low-quality evidence.
    a = _claim(
        cid="a",
        text="Vaccines are unsafe.",
        source_metadata={"role": "query"},
    )
    b = _claim(
        cid="b",
        text="Anonymous social post says vaccines are unsafe.",
        source_metadata={"role": "retrieved_evidence", "source_medium": "social_media"},
    )
    tp = type_pair(_sp(signal="conflict", stance="oppose", nli=None), a, b)
    assert tp.conflict_type == "misinformation"


def test_opinion_from_marker_words() -> None:
    a = _claim(cid="a", text="Analysts believe the economy will shrink.")
    b = _claim(cid="b", text="The economy grew 3% last year.")
    tp = type_pair(_sp(), a, b)
    assert tp.conflict_type == "opinion_conflict"
    assert tp.resolution_policy == "show_all_sides"


def test_hard_contradiction_fallback() -> None:
    a = _claim(cid="a", text="The launch succeeded.")
    b = _claim(cid="b", text="The launch did not succeed.")
    tp = type_pair(_sp(), a, b)
    assert tp.conflict_type == "hard_contradiction"
    assert tp.resolution_policy == "abstain"


def test_inconclusive_defaults_to_ambiguity() -> None:
    a = _claim(cid="a", text="Plain text.")
    b = _claim(cid="b", text="Plain text.")
    tp = type_pair(_sp(signal="inconclusive", stance="support", nli="contradiction"),
                   a, b)
    assert tp.conflict_type == "ambiguity"


def test_missing_claim_is_low_confidence_ambiguity() -> None:
    tp = type_pair(_sp(), None, _claim())
    assert tp.conflict_type == "ambiguity"
    assert tp.typing_confidence <= 0.2


def test_agreement_override_negation_mismatch() -> None:
    # Agreement can be wrong on out-of-domain snippets; contradiction
    # lexicon should recover a conflict candidate.
    a = _claim(cid="a", text="The mayor approved the budget proposal yesterday.")
    b = _claim(cid="b", text="The mayor did not approve the budget proposal yesterday.")
    tp = type_pair(_sp(signal="agreement", stance="support", nli=None), a, b)
    assert tp.conflict_type == "hard_contradiction"
    assert any("agreement/neutral signal override" in r for r in tp.rationale)


def test_agreement_not_overridden_when_nli_entailment() -> None:
    a = _claim(cid="a", text="The mayor approved the budget proposal yesterday.")
    b = _claim(cid="b", text="The mayor did not approve the budget proposal yesterday.")
    tp = type_pair(_sp(signal="agreement", stance="support", nli="entailment"), a, b)
    assert tp.conflict_type == "none"


def test_agreement_override_from_source_quality_gap() -> None:
    # In stance-only mode, a social-media query claim opposed by a
    # high-authority archived news source should not stay as plain "none".
    a = _claim(
        cid="a",
        text="Vaccines are ineffective.",
        source_metadata={"source_medium": "social_media"},
    )
    b = _claim(
        cid="b",
        text="Reuters reported vaccines significantly reduce severe disease.",
        source_metadata={
            "source_medium": "web text",
            "source_url": "https://web.archive.org/web/20220101/https://www.reuters.com/world/",
        },
    )
    tp = type_pair(_sp(signal="agreement", stance="support", nli=None), a, b)
    assert tp.conflict_type == "ambiguity"
    assert any("source quality gap" in r for r in tp.rationale)


# -------------------------------------------------------------------------
# News-domain adaptations (v0.2)
# -------------------------------------------------------------------------


def test_opinion_chinese_marker() -> None:
    a = _claim(cid="a", text="专家认为经济即将下滑。")
    b = _claim(cid="b", text="The economy grew last quarter.")
    tp = type_pair(_sp(), a, b)
    assert tp.conflict_type == "opinion_conflict"


def test_opinion_reported_speech_english() -> None:
    a = _claim(cid="a", text="Officials said the summit was a success.")
    b = _claim(cid="b", text="Attendees disputed the outcome of the summit.")
    tp = type_pair(_sp(), a, b)
    assert tp.conflict_type == "opinion_conflict"


def test_opinion_medium_is_not_misinformation() -> None:
    # An op-ed contradicting an official statement should route to
    # opinion_conflict, NOT misinformation (op-ed is subjective, not low quality).
    a = _claim(cid="a", text="Policy is failing.",
               source_metadata={"source_medium": "op_ed"})
    b = _claim(cid="b", text="Policy is succeeding.",
               source_metadata={"source_medium": "official"})
    tp = type_pair(_sp(), a, b)
    assert tp.conflict_type == "opinion_conflict"


def test_opinion_medium_detected_from_url_path() -> None:
    a = _claim(
        cid="a",
        text="The policy is harmful.",
        source_metadata={"source_url": "https://example.com/opinion/editorial/policy"},
    )
    b = _claim(
        cid="b",
        text="The policy is effective.",
        source_metadata={"source_medium": "official"},
    )
    tp = type_pair(_sp(), a, b)
    assert tp.conflict_type == "opinion_conflict"


def test_temporal_month_precision() -> None:
    # Same year, different months -> should still be temporal.
    a = _claim(cid="a", text="Rates were raised in 2023-03.", time="2023-03")
    b = _claim(cid="b", text="Rates were cut in 2023-09.", time="2023-09")
    tp = type_pair(_sp(), a, b)
    assert tp.conflict_type == "temporal_conflict"
    # rationale should mention month precision
    assert any("-03" in r and "-09" in r for r in tp.rationale)


def test_temporal_from_claim_date_metadata() -> None:
    # Year differs only in source_metadata.claim_date (not text/time)
    a = _claim(cid="a", text="Unemployment stood at high levels.",
               source_metadata={"claim_date": "2020-06-01"})
    b = _claim(cid="b", text="Unemployment has recovered significantly.",
               source_metadata={"claim_date": "2023-06-01"})
    tp = type_pair(_sp(), a, b)
    assert tp.conflict_type == "temporal_conflict"


def test_numerical_clash_routes_to_ambiguity() -> None:
    # Different headline stats, no time markers -> ambiguity (not hard_contradiction).
    a = _claim(cid="a", text="The death toll reached 100,000 nationwide.")
    b = _claim(cid="b", text="The death toll stood at 150,000 nationwide.")
    tp = type_pair(_sp(), a, b)
    assert tp.conflict_type == "ambiguity"
    assert any("numerical clash" in r for r in tp.rationale)


def test_wire_service_vs_social_media_is_misinformation() -> None:
    # New authority tier: wire_service (0.95) vs social_media (0.1)
    # -> big gap -> misinformation
    a = _claim(cid="a", text="An unverified rumor.",
               source_metadata={"source_medium": "social_media"})
    b = _claim(cid="b", text="A verified report.",
               source_metadata={"source_medium": "wire_service"})
    tp = type_pair(_sp(), a, b)
    assert tp.conflict_type == "misinformation"


def main() -> int:
    tests = [
        test_none_when_agreement,
        test_noise_when_filtered,
        test_temporal_wins_when_years_differ,
        test_ambiguity_when_subjects_differ,
        test_ambiguity_from_same_name_disambiguation,
        test_misinformation_from_medium_gap,
        test_misinformation_with_role_prior_fallback,
        test_opinion_from_marker_words,
        test_hard_contradiction_fallback,
        test_inconclusive_defaults_to_ambiguity,
        test_missing_claim_is_low_confidence_ambiguity,
        test_agreement_override_negation_mismatch,
        test_agreement_not_overridden_when_nli_entailment,
        test_agreement_override_from_source_quality_gap,
        # news-domain (v0.2)
        test_opinion_chinese_marker,
        test_opinion_reported_speech_english,
        test_opinion_medium_is_not_misinformation,
        test_opinion_medium_detected_from_url_path,
        test_temporal_month_precision,
        test_temporal_from_claim_date_metadata,
        test_numerical_clash_routes_to_ambiguity,
        test_wire_service_vs_social_media_is_misinformation,
    ]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL {t.__name__}: {e}")
    if fails:
        print(f"\n{fails} failure(s).")
        return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
