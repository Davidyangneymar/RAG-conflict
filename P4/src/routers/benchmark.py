# routers/benchmark.py
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional

from ..dependencies import get_app_state, AppState
from ..services.retrieval_client import retrieve_chunks
from ..services.p1_runner import build_retrieval_input, run_p1
from ..services.p2_runner import run_p2_with_p1
from ..services.p6_runner import run_p6_benchmark
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/benchmark", tags=["benchmark"])

class FeverClaim(BaseModel):
    id: str
    label: str
    claim: str

class BatchRequest(BaseModel):
    claims: List[FeverClaim]

@router.post("/export", response_class=PlainTextResponse)
async def export_benchmark(request: BatchRequest, state: AppState = Depends(get_app_state)):
    lines = []
    for claim_obj in request.claims:
        sample_id = str(claim_obj.id)
        query = claim_obj.claim
        gold_label = claim_obj.label

        # 1. Retrieval (uses local BM25, ignores state.retrieval_service)
        chunks = await retrieve_chunks(query, top_k=5, retrieval_service=state.retrieval_service)

        # 2. Build P1 input
        p1_input = build_retrieval_input(sample_id, query, chunks)

        # 3. Run P1
        p1_output = run_p1(p1_input, state.p1_pipeline)

        # 4. Run P2 (with p1_output)
        typed_output, input_record, p1_output = run_p2_with_p1(p1_output)

        # 5. Run P6 benchmark (no LLM)
        plan = run_p6_benchmark(typed_output, input_record, state.p6_func, p1_output)

        # 6. Build row
        row = {
            "sample_id": sample_id,
            "dataset": "fever",
            "gold_label": gold_label,
            "predicted_label": plan["predicted_label"],
            "query": query,
            "retrieved_chunk_count": len(chunks),
            "claim_count": len(p1_output.get("claims", [])),
            "candidate_pair_count": len(p1_output.get("candidate_pairs", [])),
            "cross_source_pair_count": len(p1_output.get("nli_results", [])),
            "best_entailment_score": plan["best_entailment_score"],
            "best_contradiction_score": plan["best_contradiction_score"],
            "best_neutral_score": plan["best_neutral_score"],
            "best_entailment_pair": None,
            "best_contradiction_pair": None,
            "best_neutral_pair": None,
            "claims": p1_output.get("claims", []),
            "cross_source_nli_results": p1_output.get("nli_results", []),
        }
        lines.append(json.dumps(row))

    return PlainTextResponse(content="\n".join(lines), media_type="text/plain")