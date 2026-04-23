"""
P6 contracts: controlled-generation strategy planning artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceItem:
    claim_id: str
    text: str
    stance_label: str
    conflict_type: str
    typing_confidence: float
    source_url: Optional[str] = None
    source_medium: Optional[str] = None
    time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "stance_label": self.stance_label,
            "conflict_type": self.conflict_type,
            "typing_confidence": self.typing_confidence,
            "source_url": self.source_url,
            "source_medium": self.source_medium,
            "time": self.time,
        }


@dataclass
class ConflictSummary:
    primary_conflict_type: str
    primary_resolution_policy: str
    pair_count: int
    contradiction_ratio: float
    low_confidence_ratio: float
    average_typing_confidence: float
    policy_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_conflict_type": self.primary_conflict_type,
            "primary_resolution_policy": self.primary_resolution_policy,
            "pair_count": self.pair_count,
            "contradiction_ratio": self.contradiction_ratio,
            "low_confidence_ratio": self.low_confidence_ratio,
            "average_typing_confidence": self.average_typing_confidence,
            "policy_distribution": self.policy_distribution,
        }


@dataclass
class AnswerContext:
    sample_id: str
    query: str
    evidence_clusters: Dict[str, List[EvidenceItem]]
    conflict_summary: ConflictSummary
    citations: List[Dict[str, Optional[str]]] = field(default_factory=list)
    trace_claim_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "evidence_clusters": {
                k: [item.to_dict() for item in v] for k, v in self.evidence_clusters.items()
            },
            "conflict_summary": self.conflict_summary.to_dict(),
            "citations": self.citations,
            "trace_claim_ids": self.trace_claim_ids,
        }


@dataclass
class PromptBundle:
    strategy_name: str
    stage_a_analysis_prompt: str
    stage_b_answer_prompt: str
    output_schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "stage_a_analysis_prompt": self.stage_a_analysis_prompt,
            "stage_b_answer_prompt": self.stage_b_answer_prompt,
            "output_schema": self.output_schema,
        }


@dataclass
class AbstentionDecision:
    should_abstain: bool
    reason: str
    threshold_snapshot: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_abstain": self.should_abstain,
            "reason": self.reason,
            "threshold_snapshot": self.threshold_snapshot,
        }


@dataclass
class AnswerPlan:
    sample_id: str
    answer_context: AnswerContext
    prompt_bundle: PromptBundle
    abstention: AbstentionDecision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "answer_context": self.answer_context.to_dict(),
            "prompt_bundle": self.prompt_bundle.to_dict(),
            "abstention": self.abstention.to_dict(),
        }

