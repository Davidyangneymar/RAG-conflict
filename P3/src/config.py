"""Centralized configuration loading for the retrieval layer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    """Runtime configuration shared across ingestion and retrieval modules."""

    embedding_backend: Literal["sentence_transformers", "hash"] = "sentence_transformers"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    reranker_backend: Literal["cross_encoder", "heuristic", "bge"] = "cross_encoder"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    bge_reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    bge_reranker_use_fp16: bool = True

    qdrant_collection_name: str = "conflict_aware_rag_dev"
    qdrant_path: str = "data/processed/qdrant"
    chunk_store_path: str = "data/processed/chunks.jsonl"
    bm25_store_path: str = "data/processed/bm25_corpus.json"

    chunk_size: int = 160
    chunk_overlap: int = 30
    chunking_version: str = "v2"
    sentence_aware_chunking: bool = True
    min_chunk_tokens: int = 24
    target_chunk_tokens: int = 96
    max_chunk_tokens: int = 144
    sentence_overlap: int = 1
    min_sentences_per_chunk: int = 2
    filter_fragmentary_chunks: bool = True
    enable_evidence_hygiene: bool = False
    evidence_hygiene_penalty_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    evidence_hygiene_skip_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    evidence_hygiene_extra_pool: int = 5
    evidence_hygiene_min_tokens: int = 6

    top_n_before_rerank: int = 24
    top_k_after_rerank: int = 8
    hybrid_alpha: float = Field(default=0.55, ge=0.0, le=1.0)

    max_per_source: int = 2
    max_per_doc: int = 2
    default_mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    enable_rerank: bool = True
    enable_diversify: bool = True

    fallback_embedding_dim: int = 256
    allow_model_fallback: bool = True
    log_level: str = "INFO"

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a repository-relative path into an absolute path."""
        path = Path(value)
        if path.is_absolute():
            return path
        return Path.cwd() / path


def load_retrieval_config(config_path: str | Path | None = None) -> RetrievalConfig:
    """Load retrieval settings from YAML, falling back to built-in defaults."""
    path = Path(config_path or "config/retrieval.yaml")
    if not path.exists():
        return RetrievalConfig()

    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    return RetrievalConfig.model_validate(raw_config)


@lru_cache(maxsize=1)
def get_cached_config(config_path: str | Path | None = None) -> RetrievalConfig:
    """Cache config for API/service startup."""
    return load_retrieval_config(config_path)
