"""
Standardized extension interfaces reserved for P5 and future modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .contracts import AnswerPlan


@dataclass
class QueryEnvelope:
    sample_id: str
    query: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceCluster:
    label: str
    items: List[Dict[str, Any]]


@dataclass
class AnswerPlanExchange:
    version: str
    query: QueryEnvelope
    clusters: List[EvidenceCluster]
    plan: Dict[str, Any]
    debug: Dict[str, Any] = field(default_factory=dict)


class P5FeedbackHook(Protocol):
    def on_answer_plan(self, plan: AnswerPlan) -> Dict[str, Any]:
        """Receive plan and return metric payload (e.g., abstention / consistency)."""


class DownstreamExporter(Protocol):
    def export(self, payload: AnswerPlanExchange) -> None:
        """Send normalized plan payload to downstream module/service."""


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
        query=QueryEnvelope(sample_id=plan.sample_id, query=ctx.query),
        clusters=clusters,
        plan=plan.to_dict(),
        debug=extras or {},
    )

