"""Stable handoff schemas for downstream module integration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class P1RetrievedChunk(BaseModel):
    """P3 retrieved chunk shape expected by P1 retrieval input contract."""

    chunk_id: str
    text: str
    rank: int
    retrieval_score: float
    source_url: str | None = None
    source_medium: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class P1RetrievalRecord(BaseModel):
    """Single retrieval record passed from P3 into P1."""

    sample_id: str
    query: str
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrieved_chunks: list[P1RetrievedChunk] = Field(default_factory=list)


class P1RetrievalBatch(BaseModel):
    """Batch wrapper accepted by P1 when exporting multiple records."""

    records: list[P1RetrievalRecord] = Field(default_factory=list)
