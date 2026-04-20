"""Loader dispatch helpers."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.fever_loader import load_fever_claims, load_fever_wiki_documents
from src.ingestion.generic_loader import GenericJsonlLoader
from src.schemas.documents import ClaimRecord, DocumentRecord


def load_documents(path: str | Path, loader_type: str, dataset: str) -> list[DocumentRecord]:
    """Dispatch document loading by loader type."""
    if loader_type == "fever_wiki":
        return load_fever_wiki_documents(path, dataset=dataset)
    if loader_type == "generic":
        return GenericJsonlLoader(dataset=dataset).load(path)
    raise ValueError(f"Unsupported document loader type: {loader_type}")


def load_claims(path: str | Path, loader_type: str, dataset: str = "fever") -> list[ClaimRecord]:
    """Dispatch claim loading by loader type."""
    if loader_type == "fever":
        return load_fever_claims(path, dataset=dataset)
    raise ValueError(f"Unsupported claim loader type: {loader_type}")
