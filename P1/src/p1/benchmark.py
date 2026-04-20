from __future__ import annotations

from p1.schemas import PipelineOutput, RetrievalInput


def pipeline_output_to_benchmark_record(
    output: PipelineOutput,
    retrieval_input: RetrievalInput,
) -> dict:
    """Serialize one retrieval-shaped P1 run into the P5 benchmark row contract."""
    cross_source_pairs = []
    for pair, result in zip(output.candidate_pairs, output.nli_results):
        role_a = pair.claim_a.source.metadata.get("role")
        role_b = pair.claim_b.source.metadata.get("role")
        if {role_a, role_b} != {"query", "retrieved_evidence"}:
            continue
        cross_source_pairs.append((pair, result))

    best_entailment = _best_by_score(cross_source_pairs, "entailment")
    best_contradiction = _best_by_score(cross_source_pairs, "contradiction")
    best_neutral = _best_by_score(cross_source_pairs, "neutral")
    best_overall = _best_overall(cross_source_pairs)

    return {
        "sample_id": retrieval_input.sample_id,
        "dataset": retrieval_input.metadata.get("dataset"),
        "split": retrieval_input.metadata.get("split"),
        "gold_label": retrieval_input.label,
        "query": retrieval_input.query,
        "retrieved_chunk_count": len(retrieval_input.retrieved_chunks),
        "claim_count": len(output.claims),
        "candidate_pair_count": len(output.candidate_pairs),
        "cross_source_pair_count": len(cross_source_pairs),
        "predicted_label": best_overall["label"] if best_overall else None,
        "best_entailment_score": best_entailment["score"] if best_entailment else 0.0,
        "best_contradiction_score": best_contradiction["score"] if best_contradiction else 0.0,
        "best_neutral_score": best_neutral["score"] if best_neutral else 0.0,
        "best_entailment_pair": best_entailment["pair"] if best_entailment else None,
        "best_contradiction_pair": best_contradiction["pair"] if best_contradiction else None,
        "best_neutral_pair": best_neutral["pair"] if best_neutral else None,
        "claims": [
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "source_chunk_id": claim.source_chunk_id,
                "source_role": claim.source.metadata.get("role"),
                "subject": claim.subject,
                "relation": claim.relation,
                "object": claim.object,
                "qualifier": claim.qualifier,
                "time": claim.time,
            }
            for claim in output.claims
        ],
        "cross_source_nli_results": [
            {
                "claim_a_id": result.claim_a_id,
                "claim_b_id": result.claim_b_id,
                "label": result.label.value,
                "entailment_score": result.entailment_score,
                "contradiction_score": result.contradiction_score,
                "neutral_score": result.neutral_score,
            }
            for _, result in cross_source_pairs
        ],
    }


def _best_by_score(cross_source_pairs: list[tuple], target: str) -> dict | None:
    if not cross_source_pairs:
        return None
    score_getter = {
        "entailment": lambda item: item[1].entailment_score,
        "contradiction": lambda item: item[1].contradiction_score,
        "neutral": lambda item: item[1].neutral_score,
    }[target]
    pair, result = max(cross_source_pairs, key=score_getter)
    return {
        "score": score_getter((pair, result)),
        "pair": {
            "claim_a_id": result.claim_a_id,
            "claim_b_id": result.claim_b_id,
        },
    }


def _best_overall(cross_source_pairs: list[tuple]) -> dict | None:
    if not cross_source_pairs:
        return None
    best_result = None
    best_score = -1.0
    best_label = None
    for _, result in cross_source_pairs:
        for label, score in (
            ("entailment", result.entailment_score),
            ("contradiction", result.contradiction_score),
            ("neutral", result.neutral_score),
        ):
            if score > best_score:
                best_score = score
                best_result = result
                best_label = label
    if best_result is None:
        return None
    return {
        "label": best_label,
        "score": best_score,
        "pair": {
            "claim_a_id": best_result.claim_a_id,
            "claim_b_id": best_result.claim_b_id,
        },
    }
