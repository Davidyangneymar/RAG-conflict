"""Minimal evidence diversification to preserve source coverage."""

from __future__ import annotations

from collections import Counter

from src.config import RetrievalConfig
from src.schemas.retrieval import RetrievedEvidence
from src.utils.text import parse_datetime


def _base_score(result: RetrievedEvidence) -> float:
    metadata_score = result.metadata.get("hygiene_adjusted_score")
    if metadata_score is not None:
        return float(metadata_score)
    return (
        result.score_rerank
        or result.score_hybrid
        or result.score_dense
        or result.score_sparse
        or 0.0
    )


class SourceDiversifier:
    """Greedy diversification that caps repeated sources and documents."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    def _recency_bonus(self, published_at: str | None) -> float:
        dt = parse_datetime(published_at)
        if dt is None:
            return 0.0
        return dt.timestamp() / 10_000_000_000.0

    def _can_add(
        self,
        result: RetrievedEvidence,
        *,
        source_counts: Counter[str],
        doc_counts: Counter[str],
        max_per_source: int,
    ) -> bool:
        source_key = result.source_name or result.title or result.doc_id
        if source_counts[source_key] >= max_per_source:
            return False
        if doc_counts[result.doc_id] >= self.config.max_per_doc:
            return False
        return True

    def diversify(
        self,
        candidates: list[RetrievedEvidence],
        *,
        top_k: int,
        min_unique_sources: int,
        max_per_source: int,
        prefer_recent: bool,
    ) -> list[RetrievedEvidence]:
        """Diversify top results without fully discarding relevance order."""
        if not candidates:
            return []

        ranked = sorted(
            candidates,
            key=lambda item: _base_score(item) + (self._recency_bonus(item.published_at) if prefer_recent else 0.0),
            reverse=True,
        )

        selected: list[RetrievedEvidence] = []
        selected_ids: set[str] = set()
        source_counts: Counter[str] = Counter()
        doc_counts: Counter[str] = Counter()

        if min_unique_sources > 1:
            for result in ranked:
                source_key = result.source_name or result.title or result.doc_id
                if source_key in source_counts:
                    continue
                if not self._can_add(
                    result,
                    source_counts=source_counts,
                    doc_counts=doc_counts,
                    max_per_source=max_per_source,
                ):
                    continue
                selected.append(result)
                selected_ids.add(result.chunk_id)
                source_counts[source_key] += 1
                doc_counts[result.doc_id] += 1
                if len(source_counts) >= min_unique_sources or len(selected) >= top_k:
                    break

        for result in ranked:
            if result.chunk_id in selected_ids:
                continue
            if not self._can_add(
                result,
                source_counts=source_counts,
                doc_counts=doc_counts,
                max_per_source=max_per_source,
            ):
                continue
            selected.append(result)
            source_key = result.source_name or result.title or result.doc_id
            source_counts[source_key] += 1
            doc_counts[result.doc_id] += 1
            if len(selected) >= top_k:
                break

        for rank, result in enumerate(selected, start=1):
            result.rank = rank
        return selected
