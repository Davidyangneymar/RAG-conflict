"""Core document and chunk schemas for retrieval ingestion."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """Normalized document record used by all corpus loaders."""

    doc_id: str
    dataset: str
    source_name: str | None = None
    source_url: str | None = None
    title: str | None = None
    published_at: str | None = None
    language: str | None = None
    full_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    """Chunk-level evidence unit stored in sparse and dense indexes."""

    chunk_id: str
    doc_id: str
    dataset: str
    source_name: str | None = None
    source_url: str | None = None
    title: str | None = None
    published_at: str | None = None
    text: str
    chunk_index: int
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimRecord(BaseModel):
    """Lightweight claim schema used for FEVER-style retrieval evaluation."""

    claim_id: str
    dataset: str
    query: str
    label: str | None = None
    verifiable: str | None = None
    evidence_titles: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
