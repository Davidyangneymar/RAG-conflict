"""Application-facing retrieval service."""

from __future__ import annotations

from pathlib import Path

from src.config import RetrievalConfig, load_retrieval_config
from src.retrieval.pipeline import RetrievalPipeline
from src.schemas.retrieval import RetrievalQuery, RetrievalResponse


class RetrievalService:
    """Thin service wrapper around the retrieval pipeline."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config: RetrievalConfig = load_retrieval_config(config_path)
        self.pipeline = RetrievalPipeline.from_artifacts(self.config)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        """Execute a retrieval request."""
        return self.pipeline.retrieve(query)
