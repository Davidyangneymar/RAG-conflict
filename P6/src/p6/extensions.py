"""
Standardized extension interfaces reserved for P5 and future modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from .contracts import AnswerPlan


@dataclass
class QueryEnvelope:
    sample_id: str
    query: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "metadata": self.metadata,
        }


@dataclass
class EvidenceCluster:
    label: str
    items: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "items": self.items,
        }


@dataclass
class AnswerPlanExchange:
    version: str
    query: QueryEnvelope
    clusters: List[EvidenceCluster]
    plan: Dict[str, Any]
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "query": self.query.to_dict(),
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "plan": self.plan,
            "debug": self.debug,
        }


class P5FeedbackHook(Protocol):
    def on_answer_plan(self, plan: AnswerPlan) -> Dict[str, Any]:
        """Receive plan and return metric payload (e.g., abstention / consistency)."""


class DownstreamExporter(Protocol):
    def export(self, payload: AnswerPlanExchange) -> None:
        """Send normalized plan payload to downstream module/service."""


def build_p5_feedback_payload(
    plan: AnswerPlan,
    *,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = plan.answer_context.conflict_summary
    clusters = {
        label: len(items) for label, items in plan.answer_context.evidence_clusters.items()
    }
    payload = {
        "sample_id": plan.sample_id,
        "strategy_name": plan.prompt_bundle.strategy_name,
        "primary_conflict_type": summary.primary_conflict_type,
        "primary_resolution_policy": summary.primary_resolution_policy,
        "pair_count": summary.pair_count,
        "contradiction_ratio": summary.contradiction_ratio,
        "low_confidence_ratio": summary.low_confidence_ratio,
        "average_typing_confidence": summary.average_typing_confidence,
        "policy_distribution": dict(summary.policy_distribution),
        "should_abstain": plan.abstention.should_abstain,
        "abstention_reason": plan.abstention.reason,
        "evidence_cluster_sizes": clusters,
        "trace_claim_count": len(plan.answer_context.trace_claim_ids),
    }
    if extras:
        payload["extras"] = extras
    return payload


class SimpleP5FeedbackHook:
    """Default feedback hook that emits plan-level diagnostics for P5 aggregation."""

    def on_answer_plan(self, plan: AnswerPlan) -> Dict[str, Any]:
        return build_p5_feedback_payload(plan)


class JsonlDownstreamExporter:
    """Append standardized exchange payloads to a JSONL file."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, payload: AnswerPlanExchange) -> None:
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload.to_dict(), ensure_ascii=False) + "\n")


def to_exchange_payload(
    plan: AnswerPlan,
    *,
    version: str = "p6.v1",
    extras: Optional[Dict[str, Any]] = None,
) -> AnswerPlanExchange:
    ctx = plan.answer_context
    clusters = [
        EvidenceCluster(label=label, items=[item.to_dict() for item in items])
        for label, items in ctx.evidence_clusters.items()
    ]
    return AnswerPlanExchange(
        version=version,
        query=QueryEnvelope(
            sample_id=plan.sample_id,
            query=ctx.query,
            metadata={
                "strategy_name": plan.prompt_bundle.strategy_name,
                "primary_resolution_policy": ctx.conflict_summary.primary_resolution_policy,
                "should_abstain": plan.abstention.should_abstain,
            },
        ),
        clusters=clusters,
        plan=plan.to_dict(),
        debug=extras or {},
    )
