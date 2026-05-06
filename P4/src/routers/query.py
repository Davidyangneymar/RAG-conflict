# routers/query.py
from fastapi import APIRouter, Depends
from ..dependencies import get_app_state, AppState
from ..services.retrieval_client import retrieve_chunks
from ..services.p1_runner import build_retrieval_input, run_p1
from ..services.p2_runner import run_p2
from ..services.p6_runner import run_p6
from ..models.response_models import QueryRequest, QueryResponse
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["query"])

@router.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest, state: AppState = Depends(get_app_state)):
    sample_id = str(uuid.uuid4())
    try:
        retrieval_source = None
        if state.retrieval_mode:
            if "qdrant" in state.retrieval_mode.lower():
                retrieval_source = "qdrant"
            elif "bm25" in state.retrieval_mode.lower():
                retrieval_source = "bm25"
                
        chunks = await retrieve_chunks(req.text, req.top_k, state.retrieval_service)
        p1_input = build_retrieval_input(sample_id, req.text, chunks)
        p1_output = run_p1(p1_input, state.p1_pipeline)
        typed_output, input_record = run_p2(p1_output)
        plan = run_p6(typed_output, input_record, state.p6_func)

        final_answer = plan.get("final_answer", "No answer generated")
        abstained = plan.get("abstention", {}).get("should_abstain", False)
        confidence = plan.get("confidence", 0.5)
        
        # Extract evidence texts from answer_context
        answer_ctx = plan.get("answer_context", {})
        evidence_clusters = answer_ctx.get("evidence_clusters", {})
        evidence = []
        for cluster in evidence_clusters.values():
            for item in cluster:
                text = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                if text:
                    evidence.append(text)
                    if len(evidence) >= 5:
                        break
            if len(evidence) >= 5:
                break
        # If no evidence found in clusters, fallback to citations (as before)
        if not evidence:
            citations = answer_ctx.get("citations", [])
            evidence = [c.get("source_url") or "" for c in citations]

        if typed_output.samples and typed_output.samples[0].pair_results:
            first_pair = typed_output.samples[0].pair_results[0]
            conflict_type = first_pair.conflict_type
            resolution_policy = first_pair.resolution_policy
        else:
            conflict_type = "none"
            resolution_policy = "pass_through"

        return QueryResponse(
            sample_id=sample_id,
            answer=final_answer,
            abstained=abstained,
            confidence=confidence,
            evidence=evidence,
            conflict_type=conflict_type,
            resolution_policy=resolution_policy,
            retrieval_source=retrieval_source
        )
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        user_message = _get_user_friendly_message(e)
        return QueryResponse(
            sample_id=sample_id,
            answer=user_message,
            abstained=True,
            confidence=0.0,
            evidence=[],
            conflict_type="error",
            resolution_policy="abstain"
        )

def _get_user_friendly_message(exception: Exception) -> str:
    """Return a user-friendly error message based on exception type."""
    err_str = str(exception).lower()
    # P1 / schema errors
    if "p1schemaerror" in err_str or "missing required field" in err_str:
        return "Invalid input format. Please check your request."
    # P2 stance model errors
    if "runtime config not found" in err_str or "fnc1_bert_stance_module" in err_str:
        return "Stance analysis model is not properly configured. Please contact support."
    # LLM / API key errors (including ZhipuAI)
    if "api key" in err_str or "authentication" in err_str or "zhipuai" in err_str:
        return "LLM service authentication failed. Please check API key configuration."
    # Network / timeout / connection errors
    if "timeout" in err_str or "connection" in err_str:
        return "Service temporarily unavailable. Please try again later."
    # P3 retrieval missing (non-critical, but still)
    if "chunks.jsonl" in err_str or "retrieval" in err_str:
        # This usually falls back to mock, not an error; but just in case
        return "Evidence retrieval is currently degraded. Results may be limited."
    # Default
    return "An unexpected error occurred. Please try again later."