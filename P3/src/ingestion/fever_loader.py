"""FEVER-specific loaders kept isolated from the generic pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.ingestion.generic_loader import GenericJsonlLoader
from src.schemas.documents import ClaimRecord, DocumentRecord
from src.utils.io import read_jsonl
from src.utils.text import decode_fever_title

logger = logging.getLogger(__name__)


def _extract_evidence_titles(evidence: list[Any]) -> list[str]:
    titles: list[str] = []
    for evidence_set in evidence or []:
        for item in evidence_set:
            if isinstance(item, list) and len(item) >= 3 and isinstance(item[2], str):
                title = decode_fever_title(item[2])
                if title not in titles:
                    titles.append(title)
    return titles


def load_fever_claims(path: str | Path, dataset: str = "fever") -> list[ClaimRecord]:
    """Load FEVER claim rows into a generic claim/query structure."""
    claims: list[ClaimRecord] = []
    for row in read_jsonl(path):
        claim = row.get("claim")
        claim_id = row.get("id")
        if claim_id is None or not isinstance(claim, str) or not claim.strip():
            logger.warning("Skipping malformed FEVER claim row with keys=%s", sorted(row))
            continue

        evidence_titles = _extract_evidence_titles(row.get("evidence") or [])
        claims.append(
            ClaimRecord(
                claim_id=str(claim_id),
                dataset=dataset,
                query=claim.strip(),
                label=row.get("label"),
                verifiable=row.get("verifiable"),
                evidence_titles=evidence_titles,
                metadata={
                    "evidence": row.get("evidence", []),
                    "raw_id": claim_id,
                },
            )
        )

    logger.info("Loaded %s FEVER claims from %s", len(claims), path)
    return claims


def load_fever_wiki_documents(path: str | Path, dataset: str = "fever_wiki") -> list[DocumentRecord]:
    """Load FEVER-linked Wikipedia rows through the generic document loader."""
    loader = GenericJsonlLoader(dataset=dataset, default_source_name="Wikipedia")
    return loader.load(path)
