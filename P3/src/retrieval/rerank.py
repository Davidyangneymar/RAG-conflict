"""Retrieve-then-rerank stage for higher precision evidence ranking."""

from __future__ import annotations

import logging

from src.config import RetrievalConfig
from src.schemas.retrieval import RetrievedEvidence
from src.utils.text import lexical_overlap_score

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker with a lightweight lexical fallback."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self._model = None

    def _load_cross_encoder(self):  # pragma: no cover - exercised only when model load succeeds
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.config.reranker_model_name)
            logger.info("Loaded reranker model: %s", self.config.reranker_model_name)
            return self._model
        except Exception as exc:
            if not self.config.allow_model_fallback:
                raise
            logger.warning(
                "Falling back to heuristic reranking because model failed to load: %s",
                exc,
            )
            self.config.reranker_backend = "heuristic"
            return None

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedEvidence],
        *,
        top_k: int,
    ) -> list[RetrievedEvidence]:
        """Rerank candidates while preserving the original score fields."""
        if not candidates:
            return []

        if self.config.reranker_backend == "heuristic":
            scores = [lexical_overlap_score(query, candidate.text) for candidate in candidates]
        else:
            model = self._load_cross_encoder()
            if model is None:
                scores = [lexical_overlap_score(query, candidate.text) for candidate in candidates]
            else:  # pragma: no cover - depends on optional model availability
                pairs = [(query, candidate.text) for candidate in candidates]
                scores = list(model.predict(pairs))

        reranked: list[RetrievedEvidence] = []
        for candidate, score in zip(candidates, scores):
            updated = candidate.model_copy(deep=True)
            updated.score_rerank = float(score)
            reranked.append(updated)

        reranked.sort(key=lambda item: item.score_rerank or 0.0, reverse=True)
        for rank, result in enumerate(reranked[:top_k], start=1):
            result.rank = rank
        return reranked[:top_k]
