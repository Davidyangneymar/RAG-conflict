# services/p1_runner.py
from p1.pipeline import build_pipeline, RetrievalInput
from p1.handoff import pipeline_output_to_p2_payload
import logging

logger = logging.getLogger(__name__)

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline

def build_retrieval_input(sample_id: str, query: str, retrieved_chunks: list) -> dict:
    return {
        "sample_id": sample_id,
        "query": query,
        "metadata": {"dataset": "user_query"},
        "retrieved_chunks": retrieved_chunks,
    }

def run_p1(input_data: dict, p1_pipeline):
    if p1_pipeline is None:
        p1_pipeline = get_pipeline()
    retrieval_input = RetrievalInput(
        sample_id=input_data["sample_id"],
        query=input_data["query"],
        metadata=input_data.get("metadata", {}),
        retrieved_chunks=input_data.get("retrieved_chunks", [])
    )
    result = p1_pipeline.run_retrieval_input(retrieval_input)
    # Pass sample_id explicitly to avoid None in output
    return pipeline_output_to_p2_payload(result, sample_id=input_data["sample_id"])