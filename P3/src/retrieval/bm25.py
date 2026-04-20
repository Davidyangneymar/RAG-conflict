"""Sparse retrieval using BM25 over chunked evidence units."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.schemas.documents import ChunkRecord
from src.schemas.retrieval import RetrievedEvidence
from src.utils.io import read_jsonl
from src.utils.text import simple_tokenize

logger = logging.getLogger(__name__)


def build_bm25_payload(chunks: list[ChunkRecord]) -> dict[str, Any]:
    """Create a serializable BM25 corpus payload."""
    return {
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "tokenized_corpus": [simple_tokenize(chunk.text) for chunk in chunks],
    }


def _matches_filters(chunk: ChunkRecord, filters: dict[str, Any]) -> bool:
    if not filters:
        return True
    dump = chunk.model_dump()
    metadata = chunk.metadata
    for key, expected in filters.items():
        if dump.get(key) == expected:
            continue
        if metadata.get(key) == expected:
            continue
        return False
    return True


class BM25Retriever:
    """In-memory BM25 retriever built over chunk texts."""

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = [simple_tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    @classmethod
    def from_chunk_store(cls, path: str | Path) -> "BM25Retriever":
        """Load chunk records from JSONL and rebuild the BM25 index."""
        rows = read_jsonl(path)
        chunks = [ChunkRecord.model_validate(row) for row in rows]
        return cls(chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedEvidence]:
        """Run sparse retrieval and return scored evidence rows."""
        tokenized_query = simple_tokenize(query)
        raw_scores = self.bm25.get_scores(tokenized_query)
        candidates: list[tuple[float, ChunkRecord]] = []

        for chunk, score in zip(self.chunks, raw_scores):
            if not _matches_filters(chunk, filters or {}):
                continue
            candidates.append((float(score), chunk))

        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)[:top_k]
        results: list[RetrievedEvidence] = []
        for rank, (score, chunk) in enumerate(ranked, start=1):
            results.append(
                RetrievedEvidence(
                    query=query,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    dataset=chunk.dataset,
                    source_name=chunk.source_name,
                    source_url=chunk.source_url,
                    title=chunk.title,
                    published_at=chunk.published_at,
                    text=chunk.text,
                    score_sparse=score,
                    rank=rank,
                    metadata=chunk.metadata,
                )
            )

        logger.debug("BM25 retrieved %s candidates for query=%r", len(results), query)
        return results
