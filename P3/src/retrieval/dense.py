"""Dense retrieval and embedding helpers backed by Qdrant."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import RetrievalConfig
from src.schemas.retrieval import RetrievedEvidence
from src.utils.text import stable_hash_embedding

logger = logging.getLogger(__name__)


class DenseEncoder:
    """Embedding wrapper with sentence-transformers and deterministic fallback."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self._model: Any | None = None

    def _load_sentence_transformer(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.embedding_model_name)
            logger.info("Loaded embedding model: %s", self.config.embedding_model_name)
            return self._model
        except Exception as exc:  # pragma: no cover - exercised only when model load fails
            if not self.config.allow_model_fallback:
                raise
            logger.warning(
                "Falling back to hash embeddings because embedding model failed to load: %s",
                exc,
            )
            self.config.embedding_backend = "hash"
            return None

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode texts using the configured backend."""
        if self.config.embedding_backend == "hash":
            return [stable_hash_embedding(text, self.config.fallback_embedding_dim) for text in texts]

        model = self._load_sentence_transformer()
        if model is None:
            return [stable_hash_embedding(text, self.config.fallback_embedding_dim) for text in texts]

        vectors = model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]


def build_qdrant_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    """Convert simple equality filters into Qdrant filter clauses."""
    if not filters:
        return None

    conditions = []
    for key, value in filters.items():
        payload_key = key if key != "metadata" else "metadata_json"
        conditions.append(models.FieldCondition(key=payload_key, match=models.MatchValue(value=value)))
    return models.Filter(must=conditions)


class DenseRetriever:
    """Qdrant-backed dense retriever over chunk embeddings."""

    def __init__(self, config: RetrievalConfig, encoder: DenseEncoder | None = None) -> None:
        self.config = config
        self.encoder = encoder or DenseEncoder(config)
        self.client = QdrantClient(path=str(config.resolve_path(config.qdrant_path)))

    def _query_points(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Compat wrapper for qdrant-client versions with different query APIs."""
        query_filter = build_qdrant_filter(filters)

        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.config.qdrant_collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

        response = self.client.query_points(
            collection_name=self.config.qdrant_collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return list(response.points)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedEvidence]:
        """Search dense vectors in local Qdrant."""
        query_vector = self.encoder.encode_texts([query])[0]
        points = self._query_points(query_vector, top_k=top_k, filters=filters)

        results: list[RetrievedEvidence] = []
        for rank, point in enumerate(points, start=1):
            payload = dict(point.payload or {})
            payload.pop("metadata_json", None)
            metadata = payload.pop("metadata", {}) or {}
            results.append(
                RetrievedEvidence(
                    query=query,
                    chunk_id=payload["chunk_id"],
                    doc_id=payload["doc_id"],
                    dataset=payload["dataset"],
                    source_name=payload.get("source_name"),
                    source_url=payload.get("source_url"),
                    title=payload.get("title"),
                    published_at=payload.get("published_at"),
                    text=payload["text"],
                    score_dense=float(point.score),
                    rank=rank,
                    metadata=metadata,
                )
            )
        return results
