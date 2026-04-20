"""HTTP routes for the retrieval service."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import APIRouter, Depends

from src.schemas.retrieval import RetrievalQuery, RetrievalResponse
from src.services.retrieval_service import RetrievalService

router = APIRouter()


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    """Create a shared retrieval service for API requests."""
    config_path = os.getenv("RETRIEVAL_CONFIG_PATH", "config/retrieval.yaml")
    return RetrievalService(config_path)


@router.get("/health")
def health() -> dict[str, str]:
    """Lightweight health check."""
    return {"status": "ok"}


@router.post("/retrieve", response_model=RetrievalResponse)
def retrieve(
    query: RetrievalQuery,
    service: RetrievalService = Depends(get_retrieval_service),
) -> RetrievalResponse:
    """Retrieve citation-ready evidence for a claim/query."""
    return service.retrieve(query)
