"""Generic JSONL corpus loader."""

from __future__ import annotations

import logging
from pathlib import Path

from src.ingestion.metadata import normalize_document_metadata
from src.schemas.documents import DocumentRecord
from src.utils.io import read_jsonl
from src.utils.text import normalize_whitespace

logger = logging.getLogger(__name__)


class GenericJsonlLoader:
    """Load JSONL corpora into normalized document records."""

    def __init__(self, dataset: str = "generic", default_source_name: str | None = None) -> None:
        self.dataset = dataset
        self.default_source_name = default_source_name

    def load(self, path: str | Path) -> list[DocumentRecord]:
        """Load JSONL rows using flexible field mapping."""
        records = read_jsonl(path)
        documents: list[DocumentRecord] = []

        for row in records:
            doc_id = str(row.get("doc_id") or row.get("id") or row.get("title") or "")
            full_text = row.get("full_text") or row.get("text") or row.get("content")
            if not doc_id or not isinstance(full_text, str) or not full_text.strip():
                logger.warning("Skipping malformed document row with keys=%s", sorted(row))
                continue

            metadata = normalize_document_metadata(
                row,
                dataset=self.dataset,
                default_source_name=self.default_source_name,
            )

            documents.append(
                DocumentRecord(
                    doc_id=doc_id,
                    dataset=self.dataset,
                    source_name=metadata["source_name"],
                    source_url=metadata["source_url"],
                    title=metadata["title"],
                    published_at=metadata["published_at"],
                    language=metadata["language"],
                    full_text=normalize_whitespace(full_text),
                    metadata=metadata["metadata"],
                )
            )

        logger.info("Loaded %s documents from %s", len(documents), path)
        return documents
