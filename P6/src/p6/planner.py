from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List

from .contracts import (
    AbstentionDecision,
    AnswerContext,
    AnswerPlan,
    ConflictSummary,
    EvidenceItem,
    PromptBundle,
)

_LOW_CONF_THRESHOLD = 0.45
_ABSTAIN_CONTRADICTION_RATIO = 0.5
_ABSTAIN_LOW_CONF_RATIO = 0.6
_ABSTAIN_AVG_CONF = 0.5

_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["结论", "分歧点", "证据列表", "置信度", "是否拒答"],
    "properties": {
        "结论": {"type": "string"},
        "分歧点": {"type": "array", "items": {"type": "string"}},
        "证据列表": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["claim_id", "立场", "来源", "内容摘要"],
                "properties": {
                    "claim_id": {"type": "string"},
                    "立场": {"type": "string"},
                    "来源": {"type": "string"},
                    "内容摘要": {"type": "string"},
                },
            },
        },
        "置信度": {"type": "number"},
        "是否拒答": {"type": "boolean"},
    },
}

_POLICY_STRATEGY = {
    "pass_through": "balanced_summary",
    "prefer_latest": "temporal_prefer_latest",
    "show_all_sides": "parallel_opinions",
    "disambiguate_first": "disambiguate_then_answer",
    "abstain": "abstain_with_explanation",
    "down_weight_low_quality": "quality_weighted_answer",
    "skip": "drop_noise",
}


def build_answer_plan_for_sample(typed_sample: Any, input_record: Any) -> AnswerPlan:
    context = build_answer_context(typed_sample, input_record)
    abstention = decide_abstention(context.conflict_summary)
    strategy_name = _choose_strategy(context.conflict_summary.primary_resolution_policy, abstention.should_abstain)
    prompt_bundle = build_prompt_bundle(context, strategy_name, abstention)
    return AnswerPlan(
        sample_id=typed_sample.sample_id,
        answer_context=context,
        prompt_bundle=prompt_bundle,
        abstention=abstention,
    )


def build_answer_plans(typed_samples: Iterable[Any], input_records: Iterable[Any]) -> List[AnswerPlan]:
    return [
        build_answer_plan_for_sample(ts, ir)
        for ts, ir in zip(typed_samples, input_records)
    ]


def build_answer_context(typed_sample: Any, input_record: Any) -> AnswerContext:
    query = _extract_query(input_record)
    evidence_clusters: Dict[str, List[EvidenceItem]] = {"support": [], "oppose": [], "neutral": []}
    evidence_by_claim: Dict[str, EvidenceItem] = {}
    policy_counter: Counter[str] = Counter()

    contradiction_count = 0
    low_conf_count = 0
    confidence_total = 0.0
    pair_count = len(typed_sample.pair_results)

    for typed_pair in typed_sample.pair_results:
        policy_counter[typed_pair.resolution_policy] += 1
        if typed_pair.conflict_type == "hard_contradiction":
            contradiction_count += 1
        confidence_total += typed_pair.typing_confidence
        if typed_pair.typing_confidence < _LOW_CONF_THRESHOLD:
            low_conf_count += 1

        for claim_id in (typed_pair.stance.claim_a_id, typed_pair.stance.claim_b_id):
            if claim_id in evidence_by_claim:
                continue
            claim = input_record.get_claim(claim_id)
            if claim is None:
                continue
            evidence = _claim_to_evidence(
                claim=claim,
                stance_label=typed_pair.stance.stance_label or "neutral",
                conflict_type=typed_pair.conflict_type,
                typing_confidence=typed_pair.typing_confidence,
            )
            evidence_by_claim[claim_id] = evidence
            evidence_clusters[_cluster_key(evidence.stance_label)].append(evidence)

    if pair_count == 0:
        primary_policy = "pass_through"
        primary_type = "none"
        avg_conf = 0.0
    else:
        primary_policy = policy_counter.most_common(1)[0][0] if policy_counter else "pass_through"
        primary_type = _primary_type(typed_sample)
        avg_conf = confidence_total / pair_count

    summary = ConflictSummary(
        primary_conflict_type=primary_type,
        primary_resolution_policy=primary_policy,
        pair_count=pair_count,
        contradiction_ratio=(contradiction_count / pair_count) if pair_count else 0.0,
        low_confidence_ratio=(low_conf_count / pair_count) if pair_count else 0.0,
        average_typing_confidence=avg_conf,
        policy_distribution=dict(policy_counter),
    )

    citations = []
    for evidence in evidence_by_claim.values():
        citations.append(
            {
                "claim_id": evidence.claim_id,
                "source_url": evidence.source_url,
                "source_medium": evidence.source_medium,
                "time": evidence.time,
            }
        )

    return AnswerContext(
        sample_id=typed_sample.sample_id,
        query=query,
        evidence_clusters=evidence_clusters,
        conflict_summary=summary,
        citations=citations,
        trace_claim_ids=sorted(evidence_by_claim.keys()),
    )


def decide_abstention(summary: ConflictSummary) -> AbstentionDecision:
    thresholds = {
        "contradiction_ratio": _ABSTAIN_CONTRADICTION_RATIO,
        "low_confidence_ratio": _ABSTAIN_LOW_CONF_RATIO,
        "average_typing_confidence": _ABSTAIN_AVG_CONF,
    }
    if summary.primary_resolution_policy == "abstain":
        return AbstentionDecision(
            should_abstain=True,
            reason="resolution_policy=abstain from P2 router",
            threshold_snapshot=thresholds,
        )
    if summary.contradiction_ratio >= _ABSTAIN_CONTRADICTION_RATIO:
        return AbstentionDecision(
            should_abstain=True,
            reason=f"high contradiction ratio {summary.contradiction_ratio:.2f}",
            threshold_snapshot=thresholds,
        )
    if (
        summary.low_confidence_ratio >= _ABSTAIN_LOW_CONF_RATIO
        and summary.average_typing_confidence <= _ABSTAIN_AVG_CONF
    ):
        return AbstentionDecision(
            should_abstain=True,
            reason=(
                "high low-confidence ratio "
                f"{summary.low_confidence_ratio:.2f} with avg conf {summary.average_typing_confidence:.2f}"
            ),
            threshold_snapshot=thresholds,
        )
    return AbstentionDecision(
        should_abstain=False,
        reason="abstention gate not triggered",
        threshold_snapshot=thresholds,
    )


def build_prompt_bundle(
    context: AnswerContext,
    strategy_name: str,
    abstention: AbstentionDecision,
) -> PromptBundle:
    stage_a = _build_stage_a_prompt(context, strategy_name, abstention)
    stage_b = _build_stage_b_prompt(context, strategy_name, abstention)
    return PromptBundle(
        strategy_name=strategy_name,
        stage_a_analysis_prompt=stage_a,
        stage_b_answer_prompt=stage_b,
        output_schema=_OUTPUT_SCHEMA,
    )


def _extract_query(input_record: Any) -> str:
    for claim in input_record.claims:
        role = str((claim.role or (claim.source_metadata or {}).get("role") or "")).lower()
        if role == "query":
            return claim.text
    if input_record.claims:
        return input_record.claims[0].text
    return ""


def _claim_to_evidence(claim: Any, stance_label: str, conflict_type: str, typing_confidence: float) -> EvidenceItem:
    md = claim.source_metadata or {}
    return EvidenceItem(
        claim_id=claim.claim_id,
        text=claim.text,
        stance_label=stance_label,
        conflict_type=conflict_type,
        typing_confidence=float(typing_confidence),
        source_url=md.get("source_url"),
        source_medium=md.get("source_medium"),
        time=claim.time or md.get("claim_date"),
    )


def _cluster_key(stance_label: str) -> str:
    stance = (stance_label or "").lower()
    if stance == "support":
        return "support"
    if stance == "oppose":
        return "oppose"
    return "neutral"


def _primary_type(typed_sample: Any) -> str:
    if not typed_sample.type_counts:
        return "none"
    return max(typed_sample.type_counts.items(), key=lambda kv: kv[1])[0]


def _choose_strategy(policy: str, should_abstain: bool) -> str:
    if should_abstain:
        return _POLICY_STRATEGY["abstain"]
    return _POLICY_STRATEGY.get(policy, "balanced_summary")


def _build_stage_a_prompt(
    context: AnswerContext,
    strategy_name: str,
    abstention: AbstentionDecision,
) -> str:
    return (
        "你是冲突证据分析器。只做分析，不给最终结论。\n"
        f"策略: {strategy_name}\n"
        f"问题: {context.query}\n"
        f"冲突摘要: {context.conflict_summary.to_dict()}\n"
        f"是否触发拒答门控: {abstention.should_abstain} ({abstention.reason})\n"
        "请输出：\n"
        "1) 证据簇中支持/反对/中立各自核心论点\n"
        "2) 主要分歧点与证据缺口\n"
        "3) 每个分歧点对应的可引用claim_id\n"
        "注意：禁止输出最终结论。"
    )


def _build_stage_b_prompt(
    context: AnswerContext,
    strategy_name: str,
    abstention: AbstentionDecision,
) -> str:
    policy = context.conflict_summary.primary_resolution_policy
    policy_instruction = _policy_instruction(policy, abstention.should_abstain)
    return (
        "你是受控生成器。必须严格按JSON输出，不要输出其他文本。\n"
        f"策略: {strategy_name}\n"
        f"问题: {context.query}\n"
        f"policy: {policy}\n"
        f"拒答门控: {abstention.should_abstain}, 原因: {abstention.reason}\n"
        f"证据上下文: {context.to_dict()}\n"
        f"策略指令: {policy_instruction}\n"
        "输出字段必须且仅有：结论, 分歧点, 证据列表, 置信度, 是否拒答。\n"
        "禁止融合互斥claim；证据列表必须引用claim_id与来源。"
    )


def _policy_instruction(policy: str, should_abstain: bool) -> str:
    if should_abstain or policy == "abstain":
        return "冲突无法消解时必须拒答，结论写明不确定原因并保持低置信度。"
    if policy == "prefer_latest":
        return "按时间轴优先较新证据，解释为何旧证据未被采用。"
    if policy == "show_all_sides":
        return "并列展示各立场，不要强行合并为单一确定结论。"
    if policy == "disambiguate_first":
        return "先给出可能解释或多答案，再给条件化结论。"
    if policy == "down_weight_low_quality":
        return "明确降权低可信来源并说明降权依据。"
    if policy == "skip":
        return "忽略噪声证据，仅使用相关证据作答。"
    return "给出平衡结论并保留不确定性。"

