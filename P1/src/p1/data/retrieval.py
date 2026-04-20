from __future__ import annotations

import json
from pathlib import Path

from p1.schemas import ChunkInput, RetrievalInput, RetrievedChunk


def normalize_retrieval_input(raw: dict) -> RetrievalInput:
    retrieved_chunks = [
        RetrievedChunk(
            chunk_id=str(chunk.get("chunk_id") or f"retrieved-{index + 1}"),
            text=(chunk.get("text") or "").strip(),
            rank=chunk.get("rank"),
            retrieval_score=chunk.get("retrieval_score"),
            source_url=chunk.get("source_url"),
            source_medium=chunk.get("source_medium"),
            metadata=chunk.get("metadata") or {},
        )
        for index, chunk in enumerate(raw.get("retrieved_chunks", []) or [])
        if (chunk.get("text") or "").strip()
    ]
    return RetrievalInput(
        sample_id=str(raw.get("sample_id") or raw.get("id") or "retrieval-sample"),
        query=(raw.get("query") or "").strip(),
        label=raw.get("label"),
        retrieved_chunks=retrieved_chunks,
        metadata=raw.get("metadata") or {},
    )


def read_retrieval_inputs(path: str | Path, limit: int | None = None) -> list[RetrievalInput]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if isinstance(raw.get("records"), list):
            raw_records = raw["records"]
        else:
            raw_records = [raw]
    elif isinstance(raw, list):
        raw_records = raw
    else:
        raise ValueError(f"Unsupported retrieval payload at {path}")
    if limit is not None:
        raw_records = raw_records[:limit]
    return [normalize_retrieval_input(item) for item in raw_records]


def retrieval_input_to_chunk_inputs(
    retrieval_input: RetrievalInput,
    *,
    include_query: bool = True,
) -> list[ChunkInput]:
    chunks: list[ChunkInput] = []
    if include_query:
        chunks.append(
            ChunkInput(
                doc_id=retrieval_input.sample_id,
                chunk_id="query",
                text=retrieval_input.query,
                metadata={
                    **retrieval_input.metadata,
                    "role": "query",
                    "label": retrieval_input.label,
                },
            )
        )

    for retrieved_chunk in retrieval_input.retrieved_chunks:
        chunks.append(
            ChunkInput(
                doc_id=retrieval_input.sample_id,
                chunk_id=retrieved_chunk.chunk_id,
                text=retrieved_chunk.text,
                metadata={
                    **retrieval_input.metadata,
                    **retrieved_chunk.metadata,
                    "role": "retrieved_evidence",
                    "label": retrieval_input.label,
                    "retrieval_rank": retrieved_chunk.rank,
                    "retrieval_score": retrieved_chunk.retrieval_score,
                    "source_url": retrieved_chunk.source_url,
                    "source_medium": retrieved_chunk.source_medium,
                },
            )
        )
    return chunks
