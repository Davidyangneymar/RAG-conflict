__version__ = "0.1.0"

from .p1_adapter import (
    Claim,
    CandidatePair,
    NLIResult,
    InputRecord,
    load_p1_payload,
    parse_p1_payload,
    P1SchemaError,
)
from .contracts import StancedPair, StancedSample, P2Output, AGREEMENT_SIGNALS
from .stance import PairStanceRunner, decide_claim_evidence_roles, fuse_stance_and_nli
from .conflict_typing import (
    TypedPair,
    TypedSample,
    ConflictTypedOutput,
    CONFLICT_TYPES,
    POLICIES,
    DEFAULT_POLICY_BY_TYPE,
    type_pair,
    type_sample,
)
from .pipeline import (
    run_p2_pipeline_from_path,
    run_p2_pipeline_from_records,
    run_full_p2_pipeline_from_path,
    run_full_p2_pipeline_from_records,
    run_full_p2_with_answer_plans_from_path,
    run_full_p2_with_answer_plans_from_records,
)
from .prompt_strategy import (
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
    # input adapter
    "Claim",
    "CandidatePair",
    "NLIResult",
    "InputRecord",
    "load_p1_payload",
    "parse_p1_payload",
    "P1SchemaError",
    # stance contract
    "StancedPair",
    "StancedSample",
    "P2Output",
    "AGREEMENT_SIGNALS",
    # stance
    "PairStanceRunner",
    "decide_claim_evidence_roles",
    "fuse_stance_and_nli",
    # conflict typing contract + typer
    "TypedPair",
    "TypedSample",
    "ConflictTypedOutput",
    "CONFLICT_TYPES",
    "POLICIES",
    "DEFAULT_POLICY_BY_TYPE",
    "type_pair",
    "type_sample",
    # pipelines
    "run_p2_pipeline_from_path",
    "run_p2_pipeline_from_records",
    "run_full_p2_pipeline_from_path",
    "run_full_p2_pipeline_from_records",
    "run_full_p2_with_answer_plans_from_path",
    "run_full_p2_with_answer_plans_from_records",
    # p6-facing prompt strategy contracts + planners
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
    # p6 extension channels (for P5/future modules)
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
