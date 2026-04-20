"""Build local sparse and dense retrieval artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import RetrievalConfig
from src.ingestion.splitter import TextSplitter
from src.retrieval.bm25 import build_bm25_payload
from src.retrieval.dense import DenseEncoder
from src.schemas.documents import ChunkRecord, DocumentRecord
from src.utils.io import ensure_dir, write_json, write_jsonl

logger = logging.getLogger(__name__)


class IndexBuilder:
    """Create chunk, BM25, and Qdrant artifacts from document records."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self.splitter = TextSplitter.from_config(config)
        self.encoder = DenseEncoder(config)

    def build_chunks(self, documents: list[DocumentRecord]) -> list[ChunkRecord]:
        """Chunk all input documents into retrieval units."""
        chunks: list[ChunkRecord] = []
        for document in documents:
            chunks.extend(self.splitter.split_document(document))
        logger.info("Created %s chunks from %s documents", len(chunks), len(documents))
        return chunks

    def _write_chunk_store(self, chunks: list[ChunkRecord]) -> Path:
        chunk_store_path = self.config.resolve_path(self.config.chunk_store_path)
        write_jsonl(chunk_store_path, (chunk.model_dump() for chunk in chunks))
        return chunk_store_path

    def _write_bm25_store(self, chunks: list[ChunkRecord]) -> Path:
        bm25_payload = build_bm25_payload(chunks)
        bm25_store_path = self.config.resolve_path(self.config.bm25_store_path)
        write_json(bm25_store_path, bm25_payload)
        return bm25_store_path

    def _write_qdrant_store(self, chunks: list[ChunkRecord]) -> Path:
        qdrant_path = self.config.resolve_path(self.config.qdrant_path)
        ensure_dir(qdrant_path)

        embeddings = self.encoder.encode_texts([chunk.text for chunk in chunks])
        vector_size = len(embeddings[0]) if embeddings else self.config.fallback_embedding_dim
        client = QdrantClient(path=str(qdrant_path))

        if client.collection_exists(self.config.qdrant_collection_name):
            client.delete_collection(self.config.qdrant_collection_name)

        client.create_collection(
            collection_name=self.config.qdrant_collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            payload = chunk.model_dump()
            payload["metadata_json"] = json.dumps(payload.get("metadata", {}), ensure_ascii=False)
            points.append(models.PointStruct(id=index, vector=vector, payload=payload))

        if points:
            client.upsert(collection_name=self.config.qdrant_collection_name, points=points)

        logger.info(
            "Indexed %s vectors into local Qdrant collection=%s",
            len(points),
            self.config.qdrant_collection_name,
        )
        return qdrant_path

    def build(self, documents: list[DocumentRecord]) -> list[ChunkRecord]:
        """Build all retrieval artifacts and return the chunk store."""
        chunks = self.build_chunks(documents)
        self._write_chunk_store(chunks)
        self._write_bm25_store(chunks)
        self._write_qdrant_store(chunks)
        return chunks
