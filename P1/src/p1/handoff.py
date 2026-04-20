from __future__ import annotations

from p1.schemas import PipelineOutput


def pipeline_output_to_p2_payload(output: PipelineOutput, *, sample_id: str | None = None) -> dict:
    """Serialize P1 output into the stable P2-facing handoff contract."""
    return {
        "sample_id": sample_id,
        "claims": [
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "source_doc_id": claim.source_doc_id,
                "source_chunk_id": claim.source_chunk_id,
                "entities": claim.entities,
                "subject": claim.subject,
                "relation": claim.relation,
                "object": claim.object,
                "qualifier": claim.qualifier,
                "time": claim.time,
                "polarity": claim.polarity,
                "certainty": claim.certainty,
                "metadata": claim.metadata,
                "source_metadata": claim.source.metadata,
            }
            for claim in output.claims
        ],
        "candidate_pairs": [
            {
                "claim_a_id": pair.claim_a.claim_id,
                "claim_b_id": pair.claim_b.claim_id,
                "entity_overlap": pair.entity_overlap,
                "lexical_similarity": pair.lexical_similarity,
                "embedding_similarity": pair.embedding_similarity,
            }
            for pair in output.candidate_pairs
        ],
        "nli_results": [
            {
                "claim_a_id": result.claim_a_id,
                "claim_b_id": result.claim_b_id,
                "label": result.label.value,
                "entailment_score": result.entailment_score,
                "contradiction_score": result.contradiction_score,
                "neutral_score": result.neutral_score,
                "is_bidirectional": result.is_bidirectional,
                "metadata": result.metadata,
            }
            for result in output.nli_results
        ],
    }
