"""Retrieval request and response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    """Stable API contract for downstream retrieval calls."""

    query: str
    top_k: int = 8
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    use_rerank: bool = True
    use_diversify: bool = True
    prefer_recent: bool = False
    min_unique_sources: int = 2
    max_per_source: int = 2
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievedEvidence(BaseModel):
    """Retrieved evidence with citation metadata and stage-wise scores."""

    query: str
    chunk_id: str
    doc_id: str
    dataset: str
    source_name: str | None = None
    source_url: str | None = None
    title: str | None = None
    published_at: str | None = None
    text: str
    score_sparse: float | None = None
    score_dense: float | None = None
    score_hybrid: float | None = None
    score_rerank: float | None = None
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    """Response envelope returned by the API and CLI scripts."""

    query: str
    mode: str
    results: list[RetrievedEvidence]
