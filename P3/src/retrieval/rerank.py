"""Retrieve-then-rerank stage for higher precision evidence ranking."""

from __future__ import annotations

import logging

from src.config import RetrievalConfig
from src.schemas.retrieval import RetrievedEvidence
from src.utils.text import lexical_overlap_score

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker with BGE and lightweight lexical fallback."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self._model = None
        self._bge_model = None

    def _load_cross_encoder(self):  # pragma: no cover
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

    def _load_bge_reranker(self):
        if self._bge_model is not None:
            return self._bge_model

        try:
            from FlagEmbedding import FlagReranker

            self._bge_model = FlagReranker(
                self.config.bge_reranker_model_name,
                use_fp16=self.config.bge_reranker_use_fp16,
            )
            logger.info("Loaded BGE reranker: %s", self.config.bge_reranker_model_name)
            return self._bge_model
        except Exception as exc:
            if not self.config.allow_model_fallback:
                raise
            logger.warning(
                "BGE reranker failed to load, falling back to heuristic: %s", exc
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

        pre_rerank_ranks = {i: rank for i, rank in enumerate(range(1, len(candidates) + 1))}

        requested_backend = self.config.reranker_backend
        fallback_reason: str | None = None

        if self.config.reranker_backend == "heuristic":
            scores = [lexical_overlap_score(query, c.text) for c in candidates]
        elif self.config.reranker_backend == "bge":
            model = self._load_bge_reranker()
            if model is None:
                fallback_reason = "bge_load_failed"
                scores = [lexical_overlap_score(query, c.text) for c in candidates]
            else:
                try:
                    pairs = [[query, c.text] for c in candidates]
                    raw_scores = model.compute_score(pairs, normalize=True)
                    if isinstance(raw_scores, (int, float)):
                        scores = [float(raw_scores)]
                    else:
                        scores = [float(s) for s in raw_scores]
                except Exception as exc:
                    if not self.config.allow_model_fallback:
                        raise
                    logger.warning("BGE reranker inference failed, falling back to heuristic: %s", exc)
                    self.config.reranker_backend = "heuristic"
                    fallback_reason = "bge_inference_failed"
                    scores = [lexical_overlap_score(query, c.text) for c in candidates]
        else:
            model = self._load_cross_encoder()
            if model is None:
                scores = [lexical_overlap_score(query, c.text) for c in candidates]
            else:  # pragma: no cover
                pairs = [(query, c.text) for c in candidates]
                scores = list(model.predict(pairs))

        if fallback_reason is not None:
            meta_extra = {
                "reranker_backend": "heuristic_fallback",
                "reranker_model": "heuristic",
                "requested_reranker_model": self.config.bge_reranker_model_name,
                "fallback_reason": fallback_reason,
            }
        elif requested_backend == "bge":
            meta_extra = {
                "reranker_backend": "bge",
                "reranker_model": self.config.bge_reranker_model_name,
            }
        elif requested_backend == "cross_encoder":
            meta_extra = {
                "reranker_backend": "cross_encoder",
                "reranker_model": self.config.reranker_model_name,
            }
        else:
            meta_extra = {
                "reranker_backend": "heuristic",
                "reranker_model": "heuristic",
            }

        reranked: list[RetrievedEvidence] = []
        for idx, (candidate, score) in enumerate(zip(candidates, scores)):
            updated = candidate.model_copy(deep=True)
            updated.score_rerank = float(score)
            updated.metadata = {
                **updated.metadata,
                "pre_rerank_rank": pre_rerank_ranks[idx],
                "rerank_score": float(score),
                **meta_extra,
            }
            reranked.append(updated)

        reranked.sort(key=lambda item: item.score_rerank or 0.0, reverse=True)
        for rank, result in enumerate(reranked[:top_k], start=1):
            result.rank = rank
        return reranked[:top_k]
