"""End-to-end retrieval pipeline orchestration."""

from __future__ import annotations

from src.config import RetrievalConfig
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseEncoder, DenseRetriever
from src.retrieval.diversify import SourceDiversifier
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rerank import Reranker
from src.schemas.retrieval import RetrievalQuery, RetrievalResponse


class RetrievalPipeline:
    """Hybrid-first retrieval pipeline for citation-ready evidence search."""

    def __init__(
        self,
        config: RetrievalConfig,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        diversifier: SourceDiversifier,
    ) -> None:
        self.config = config
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.diversifier = diversifier

    @classmethod
    def from_artifacts(cls, config: RetrievalConfig) -> "RetrievalPipeline":
        """Build a pipeline from persisted chunk and vector artifacts."""
        bm25_retriever = BM25Retriever.from_chunk_store(config.resolve_path(config.chunk_store_path))
        encoder = DenseEncoder(config)
        dense_retriever = DenseRetriever(config, encoder=encoder)
        hybrid_retriever = HybridRetriever(
            config=config,
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
        )
        reranker = Reranker(config)
        diversifier = SourceDiversifier(config)
        return cls(
            config=config,
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            diversifier=diversifier,
        )

    def retrieve(self, request: RetrievalQuery) -> RetrievalResponse:
        """Retrieve balanced evidence for a claim/query."""
        candidate_k = max(request.top_k * 3, self.config.top_n_before_rerank)

        if request.mode == "bm25":
            candidates = self.bm25_retriever.search(
                request.query,
                top_k=candidate_k,
                filters=request.filters,
            )
        elif request.mode == "dense":
            candidates = self.dense_retriever.search(
                request.query,
                top_k=candidate_k,
                filters=request.filters,
            )
        else:
            candidates = self.hybrid_retriever.search(
                request.query,
                top_k=candidate_k,
                filters=request.filters,
            )

        if request.use_rerank and self.config.enable_rerank:
            candidates = self.reranker.rerank(request.query, candidates, top_k=candidate_k)

        if request.use_diversify and self.config.enable_diversify:
            candidates = self.diversifier.diversify(
                candidates,
                top_k=request.top_k,
                min_unique_sources=request.min_unique_sources,
                max_per_source=request.max_per_source,
                prefer_recent=request.prefer_recent,
            )
        else:
            candidates = candidates[: request.top_k]

        candidates = candidates[: request.top_k]

        for rank, result in enumerate(candidates, start=1):
            result.rank = rank

        return RetrievalResponse(query=request.query, mode=request.mode, results=candidates)
