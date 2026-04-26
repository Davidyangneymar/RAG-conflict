"""
Compatibility bridge: keep legacy `src.p2.prompt_strategy` API stable while
the implementation lives in the standalone root-level P6 module.
"""

from __future__ import annotations

from ._bridge import ensure_p6_on_path

ensure_p6_on_path()

from p6 import (  # type: ignore  # noqa: E402
    EvidenceItem,
    ConflictSummary,
    AnswerContext,
    PromptBundle,
    AbstentionDecision,
    AnswerPlan,
    build_answer_context,
    decide_abstention,
    build_prompt_bundle,
    build_answer_plan_for_sample,
    build_answer_plans,
    QueryEnvelope,
    EvidenceCluster,
    AnswerPlanExchange,
    P5FeedbackHook,
    DownstreamExporter,
    build_p5_feedback_payload,
    SimpleP5FeedbackHook,
    JsonlDownstreamExporter,
    to_exchange_payload,
)

__all__ = [
    "EvidenceItem",
    "ConflictSummary",
    "AnswerContext",
    "PromptBundle",
    "AbstentionDecision",
    "AnswerPlan",
    "build_answer_context",
    "decide_abstention",
    "build_prompt_bundle",
    "build_answer_plan_for_sample",
    "build_answer_plans",
    "QueryEnvelope",
    "EvidenceCluster",
    "AnswerPlanExchange",
    "P5FeedbackHook",
    "DownstreamExporter",
    "build_p5_feedback_payload",
    "SimpleP5FeedbackHook",
    "JsonlDownstreamExporter",
    "to_exchange_payload",
]
