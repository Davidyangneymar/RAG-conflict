"""Metadata normalization helpers for corpus ingestion."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from src.utils.text import decode_fever_title


def build_wikipedia_url(title: str | None) -> str | None:
    """Build a best-effort Wikipedia URL from a normalized title."""
    if not title:
        return None
    slug = quote(title.replace(" ", "_"), safe="_()")
    return f"https://en.wikipedia.org/wiki/{slug}"


def normalize_document_metadata(
    raw: dict[str, Any],
    *,
    dataset: str,
    default_source_name: str | None = None,
) -> dict[str, Any]:
    """Extract common metadata fields while preserving unknown extras."""
    used_keys = {
        "id",
        "doc_id",
        "title",
        "text",
        "full_text",
        "content",
        "source_name",
        "source_url",
        "published_at",
        "language",
        "metadata",
    }

    title = raw.get("title") or raw.get("id")
    if isinstance(title, str):
        title = decode_fever_title(title)

    source_name = raw.get("source_name") or default_source_name
    if not source_name and "wiki" in dataset.lower():
        source_name = "Wikipedia"

    source_url = raw.get("source_url")
    if not source_url and source_name == "Wikipedia":
        source_url = build_wikipedia_url(title)

    metadata = dict(raw.get("metadata") or {})
    for key, value in raw.items():
        if key not in used_keys:
            metadata[key] = value

    return {
        "source_name": source_name,
        "source_url": source_url,
        "title": title,
        "published_at": raw.get("published_at"),
        "language": raw.get("language"),
        "metadata": metadata,
    }
