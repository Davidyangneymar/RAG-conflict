"""Adapters that convert P3 retrieval outputs into P1 contract payloads."""

from __future__ import annotations

from typing import Any

from src.schemas.documents import ClaimRecord
from src.schemas.handoff import P1RetrievalBatch, P1RetrievalRecord, P1RetrievedChunk
from src.schemas.retrieval import RetrievalResponse, RetrievedEvidence
from src.services.evidence_hygiene import select_retrieval_score


def evidence_to_p1_chunk(evidence: RetrievedEvidence) -> P1RetrievedChunk:
    """Convert a retrieved evidence item into the P1 chunk contract.

    Mapping note:
    - P3 uses `source_name`
    - P1 contract expects `source_medium`
    We intentionally map `source_name -> source_medium` here.
    """
    metadata: dict[str, Any] = {
        "title": evidence.title,
        "source_doc_id": evidence.doc_id,
        "dataset": evidence.dataset,
        "source_name": evidence.source_name,
        "published_at": evidence.published_at,
    }
    metadata.update(evidence.metadata)

    return P1RetrievedChunk(
        chunk_id=evidence.chunk_id,
        text=evidence.text,
        rank=evidence.rank,
        retrieval_score=select_retrieval_score(evidence),
        source_url=evidence.source_url,
        source_medium=evidence.source_name,
        metadata=metadata,
    )


def retrieval_response_to_p1_record(
    response: RetrievalResponse,
    *,
    sample_id: str,
    label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> P1RetrievalRecord:
    """Convert a retrieval response into a single P1 handoff record."""
    return P1RetrievalRecord(
        sample_id=sample_id,
        query=response.query,
        label=label,
        metadata=metadata or {},
        retrieved_chunks=[evidence_to_p1_chunk(result) for result in response.results],
    )


def claim_and_response_to_p1_record(
    claim: ClaimRecord,
    response: RetrievalResponse,
    *,
    split: str | None = None,
) -> P1RetrievalRecord:
    """Build a P1 handoff record from a FEVER-style claim and retrieval output."""
    metadata = {
        "dataset": claim.dataset,
    }
    if split:
        metadata["split"] = split

    claim_date = claim.metadata.get("claim_date")
    if claim_date is not None:
        metadata["claim_date"] = claim_date

    return retrieval_response_to_p1_record(
        response,
        sample_id=claim.claim_id,
        label=claim.label,
        metadata=metadata,
    )


def responses_to_p1_batch(records: list[P1RetrievalRecord]) -> P1RetrievalBatch:
    """Wrap multiple handoff records in the batch shape accepted by P1."""
    return P1RetrievalBatch(records=records)
