from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SchemaMixin:
    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class NliLabel(str, Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


@dataclass
class ClaimSource(SchemaMixin):
    doc_id: str
    chunk_id: str | None = None
    sentence_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Claim(SchemaMixin):
    claim_id: str
    text: str
    source: ClaimSource
    source_doc_id: str | None = None
    source_chunk_id: str | None = None
    entities: list[str] = field(default_factory=list)
    subject: str | None = None
    relation: str | None = None
    object: str | None = None
    qualifier: str | None = None
    time: str | None = None
    polarity: str = "uncertain"
    certainty: float = 0.5
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_doc_id is None:
            self.source_doc_id = self.source.doc_id
        if self.source_chunk_id is None:
            self.source_chunk_id = self.source.chunk_id


@dataclass
class ClaimPair(SchemaMixin):
    claim_a: Claim
    claim_b: Claim
    entity_overlap: list[str] = field(default_factory=list)
    lexical_similarity: float = 0.0
    embedding_similarity: float | None = None


@dataclass
class NLIPairResult(SchemaMixin):
    claim_a_id: str
    claim_b_id: str
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    label: NliLabel
    is_bidirectional: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkInput(SchemaMixin):
    doc_id: str
    text: str
    chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk(SchemaMixin):
    chunk_id: str
    text: str
    rank: int | None = None
    retrieval_score: float | None = None
    source_url: str | None = None
    source_medium: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalInput(SchemaMixin):
    sample_id: str
    query: str
    label: str | None = None
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineOutput(SchemaMixin):
    claims: list[Claim] = field(default_factory=list)
    candidate_pairs: list[ClaimPair] = field(default_factory=list)
    nli_results: list[NLIPairResult] = field(default_factory=list)
