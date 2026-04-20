"""Hybrid retrieval combining sparse and dense scores."""

from __future__ import annotations

from src.config import RetrievalConfig
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.schemas.retrieval import RetrievedEvidence


def _normalize_scores(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    min_score = min(values.values())
    max_score = max(values.values())
    if max_score == min_score:
        return {key: 1.0 for key in values}
    return {key: (score - min_score) / (max_score - min_score) for key, score in values.items()}


class HybridRetriever:
    """Weighted sparse+dense fusion for evidence retrieval."""

    def __init__(
        self,
        config: RetrievalConfig,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
    ) -> None:
        self.config = config
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedEvidence]:
        """Run both retrievers, normalize scores, and combine them."""
        sparse_results = self.bm25_retriever.search(query, top_k=top_k, filters=filters)
        dense_results = self.dense_retriever.search(query, top_k=top_k, filters=filters)

        sparse_scores = _normalize_scores(
            {result.chunk_id: result.score_sparse or 0.0 for result in sparse_results}
        )
        dense_scores = _normalize_scores(
            {result.chunk_id: result.score_dense or 0.0 for result in dense_results}
        )

        merged: dict[str, RetrievedEvidence] = {}
        for result in sparse_results + dense_results:
            current = merged.get(result.chunk_id)
            if current is None:
                merged[result.chunk_id] = result.model_copy(deep=True)
                continue
            if result.score_sparse is not None:
                current.score_sparse = result.score_sparse
            if result.score_dense is not None:
                current.score_dense = result.score_dense

        for chunk_id, result in merged.items():
            sparse_score = sparse_scores.get(chunk_id, 0.0)
            dense_score = dense_scores.get(chunk_id, 0.0)
            result.score_hybrid = (self.config.hybrid_alpha * dense_score) + (
                (1.0 - self.config.hybrid_alpha) * sparse_score
            )

        ranked = sorted(merged.values(), key=lambda item: item.score_hybrid or 0.0, reverse=True)[:top_k]
        for rank, result in enumerate(ranked, start=1):
            result.rank = rank
        return ranked
