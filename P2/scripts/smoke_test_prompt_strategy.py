"""
Pure-Python smoke tests for the P6-facing prompt strategy planner.

Run:
    python scripts/smoke_test_prompt_strategy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.p2 import (  # noqa: E402
    Claim,
    InputRecord,
    StancedPair,
    TypedPair,
    TypedSample,
    build_answer_plan_for_sample,
)


def _record() -> InputRecord:
    claims = [
        Claim(
            claim_id="Q",
            text="The policy worked or failed?",
            role="query",
            source_metadata={"role": "query"},
        ),
        Claim(
            claim_id="E1",
            text="Reuters says policy succeeded in 2025.",
            time="2025",
            source_metadata={
                "role": "retrieved_evidence",
                "source_url": "https://reuters.com/a",
                "source_medium": "wire_service",
            },
        ),
        Claim(
            claim_id="E2",
            text="Blog says policy failed in 2024.",
            time="2024",
            source_metadata={
                "role": "retrieved_evidence",
                "source_url": "https://example.blog/b",
                "source_medium": "blog",
            },
        ),
    ]
    return InputRecord(
        sample_id="s1",
        claims=claims,
        candidate_pairs=[],
        nli_results=[],
        _claims_by_id={c.claim_id: c for c in claims},
    )


def _stanced(a: str, b: str, stance_label: str = "oppose") -> StancedPair:
    return StancedPair(
        claim_a_id=a,
        claim_b_id=b,
        nli_label="contradiction",
        stance_label=stance_label,
        stance_decision_score=0.8,
        stance_direction="a_as_claim",
        is_filtered=False,
        agreement_signal="conflict",
        fusion_confidence=0.8,
        notes=[],
    )


def _typed_pair(policy: str, conflict_type: str, conf: float, stance_label: str = "oppose") -> TypedPair:
    return TypedPair(
        stance=_stanced("Q", "E1", stance_label=stance_label),
        conflict_type=conflict_type,
        typing_confidence=conf,
        resolution_policy=policy,
        rationale=["test rationale"],
    )


def test_prefer_latest_strategy() -> None:
    ts = TypedSample(
        sample_id="s1",
        pair_results=[_typed_pair("prefer_latest", "temporal_conflict", 0.9)],
        type_counts={"temporal_conflict": 1},
    )
    plan = build_answer_plan_for_sample(ts, _record())
    assert plan.prompt_bundle.strategy_name == "temporal_prefer_latest"
    assert "按时间轴优先较新证据" in plan.prompt_bundle.stage_b_answer_prompt
    assert plan.answer_context.query == "The policy worked or failed?"


def test_show_all_sides_strategy() -> None:
    ts = TypedSample(
        sample_id="s1",
        pair_results=[_typed_pair("show_all_sides", "opinion_conflict", 0.85)],
        type_counts={"opinion_conflict": 1},
    )
    plan = build_answer_plan_for_sample(ts, _record())
    assert plan.prompt_bundle.strategy_name == "parallel_opinions"
    assert "并列展示各立场" in plan.prompt_bundle.stage_b_answer_prompt
    assert not plan.abstention.should_abstain


def test_abstain_policy_forces_abstention() -> None:
    ts = TypedSample(
        sample_id="s1",
        pair_results=[_typed_pair("abstain", "hard_contradiction", 0.8)],
        type_counts={"hard_contradiction": 1},
    )
    plan = build_answer_plan_for_sample(ts, _record())
    assert plan.abstention.should_abstain
    assert plan.prompt_bundle.strategy_name == "abstain_with_explanation"


def test_contradiction_ratio_gate() -> None:
    ts = TypedSample(
        sample_id="s1",
        pair_results=[
            _typed_pair("pass_through", "hard_contradiction", 0.7),
            _typed_pair("pass_through", "hard_contradiction", 0.7),
            _typed_pair("pass_through", "none", 0.8),
        ],
        type_counts={"hard_contradiction": 2, "none": 1},
    )
    plan = build_answer_plan_for_sample(ts, _record())
    assert plan.abstention.should_abstain
    assert "high contradiction ratio" in plan.abstention.reason


def test_output_schema_fields() -> None:
    ts = TypedSample(
        sample_id="s1",
        pair_results=[_typed_pair("pass_through", "none", 0.9, stance_label="support")],
        type_counts={"none": 1},
    )
    plan = build_answer_plan_for_sample(ts, _record())
    required = plan.prompt_bundle.output_schema["required"]
    assert required == ["结论", "分歧点", "证据列表", "置信度", "是否拒答"]
    assert "证据列表" in plan.prompt_bundle.stage_b_answer_prompt
    assert "Q" in plan.answer_context.trace_claim_ids


def main() -> int:
    tests = [
        test_prefer_latest_strategy,
        test_show_all_sides_strategy,
        test_abstain_policy_forces_abstention,
        test_contradiction_ratio_gate,
        test_output_schema_fields,
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

