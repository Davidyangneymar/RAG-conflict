"""AVeriTeC dev-only smoke loaders.

These helpers intentionally adapt only the lightweight development JSON shape.
They do not ingest the full AVeriTeC knowledge store.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.schemas.documents import ClaimRecord, DocumentRecord
from src.utils.io import read_jsonl
from src.utils.text import normalize_whitespace

logger = logging.getLogger(__name__)


NO_ANSWER_TEXT = "no answer could be found"


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load AVeriTeC rows from the official dev JSON or derived JSONL files."""
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        return read_jsonl(source)

    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        records = data.get("data") or data.get("records") or []
        return [row for row in records if isinstance(row, dict)]
    raise ValueError(f"Unsupported AVeriTeC dev file shape: {type(data).__name__}")


def _claim_id(index: int, row: dict[str, Any]) -> str:
    raw_id = row.get("claim_id") or row.get("id") or row.get("uid")
    return str(raw_id) if raw_id is not None else f"averitec_dev_{index}"


def _base_metadata(index: int, row: dict[str, Any]) -> dict[str, Any]:
    """Preserve claim-level fields needed by downstream diagnostics."""
    return {
        "claim_id": _claim_id(index, row),
        "claim": row.get("claim"),
        "label": row.get("label"),
        "claim_date": row.get("claim_date"),
        "speaker": row.get("speaker"),
        "publisher": row.get("publisher") or row.get("reporting_source"),
        "reporting_source": row.get("reporting_source"),
        "location": row.get("location") or row.get("location_ISO_code"),
        "original_claim_url": row.get("original_claim_url"),
        "cached_original_claim_url": row.get("cached_original_claim_url"),
        "fact_checking_article": row.get("fact_checking_article"),
        "claim_types": row.get("claim_types") or [],
        "fact_checking_strategies": row.get("fact_checking_strategies") or [],
        "required_reannotation": row.get("required_reannotation"),
        "row_index": index,
    }


def _iter_answer_documents(index: int, row: dict[str, Any]) -> list[DocumentRecord]:
    claim_id = _claim_id(index, row)
    metadata = _base_metadata(index, row)
    documents: list[DocumentRecord] = []

    for question_index, question_row in enumerate(row.get("questions") or []):
        if not isinstance(question_row, dict):
            continue
        question = normalize_whitespace(str(question_row.get("question") or ""))
        for answer_index, answer_row in enumerate(question_row.get("answers") or []):
            if not isinstance(answer_row, dict):
                continue

            answer = normalize_whitespace(str(answer_row.get("answer") or ""))
            if not answer or answer.lower().strip(". ") == NO_ANSWER_TEXT:
                continue

            answer_metadata = {
                **metadata,
                "source_type": "qa_answer",
                "question": question,
                "answer": answer,
                "answer_type": answer_row.get("answer_type"),
                "question_index": question_index,
                "answer_index": answer_index,
                "cached_source_url": answer_row.get("cached_source_url"),
            }
            source_name = answer_row.get("source_medium") or metadata.get("publisher") or "AVeriTeC QA answer"
            source_url = answer_row.get("source_url") or answer_row.get("cached_source_url")
            text = normalize_whitespace(f"Question: {question}\nAnswer: {answer}")

            documents.append(
                DocumentRecord(
                    doc_id=f"{claim_id}::q{question_index}::a{answer_index}",
                    dataset="averitec_dev_smoke",
                    source_name=source_name,
                    source_url=source_url or None,
                    title=question or f"AVeriTeC claim {claim_id}",
                    published_at=row.get("claim_date"),
                    language="en",
                    full_text=text,
                    metadata=answer_metadata,
                )
            )
    return documents


def _justification_document(index: int, row: dict[str, Any]) -> DocumentRecord | None:
    justification = normalize_whitespace(str(row.get("justification") or ""))
    claim = normalize_whitespace(str(row.get("claim") or ""))
    if not justification:
        return None

    claim_id = _claim_id(index, row)
    metadata = {
        **_base_metadata(index, row),
        "source_type": "justification",
        "justification": justification,
    }
    text = normalize_whitespace(f"Claim: {claim}\nFact-check justification: {justification}")

    return DocumentRecord(
        doc_id=f"{claim_id}::justification",
        dataset="averitec_dev_smoke",
        source_name=row.get("reporting_source") or "AVeriTeC justification",
        source_url=row.get("fact_checking_article"),
        title=f"AVeriTeC justification for {claim_id}",
        published_at=row.get("claim_date"),
        language="en",
        full_text=text,
        metadata=metadata,
    )


def load_averitec_dev_documents(path: str | Path, dataset: str = "averitec_dev_smoke") -> list[DocumentRecord]:
    """Load AVeriTeC dev QA answers/justifications as smoke corpus documents."""
    documents: list[DocumentRecord] = []
    for index, row in enumerate(_load_rows(path)):
        for document in _iter_answer_documents(index, row):
            documents.append(document.model_copy(update={"dataset": dataset}))
        justification = _justification_document(index, row)
        if justification is not None:
            documents.append(justification.model_copy(update={"dataset": dataset}))

    logger.info("Loaded %s AVeriTeC dev smoke documents from %s", len(documents), path)
    return documents


def load_averitec_dev_claims(path: str | Path, dataset: str = "averitec_dev") -> list[ClaimRecord]:
    """Load AVeriTeC dev claims into the generic retrieval-query schema."""
    claims: list[ClaimRecord] = []
    for index, row in enumerate(_load_rows(path)):
        claim = normalize_whitespace(str(row.get("claim") or ""))
        if not claim:
            logger.warning("Skipping AVeriTeC row without claim at index=%s", index)
            continue

        metadata = _base_metadata(index, row)
        metadata["justification"] = row.get("justification")
        metadata["questions"] = row.get("questions") or []

        claims.append(
            ClaimRecord(
                claim_id=_claim_id(index, row),
                dataset=dataset,
                query=claim,
                label=row.get("label"),
                verifiable=None,
                evidence_titles=[],
                metadata=metadata,
            )
        )

    logger.info("Loaded %s AVeriTeC dev smoke claims from %s", len(claims), path)
    return claims
