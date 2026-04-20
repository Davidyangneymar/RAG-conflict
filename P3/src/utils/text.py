"""Text normalization helpers for loaders and retrieval."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime
from typing import Iterable


FEVER_MARKUP_MAP = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
    "``": '"',
    "''": '"',
}


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def decode_fever_title(text: str) -> str:
    """Convert FEVER/Wikipedia title tokens into human-readable form."""
    normalized = text.replace("_", " ")
    for old, new in FEVER_MARKUP_MAP.items():
        normalized = normalized.replace(old, new)
    return normalize_whitespace(normalized)


def simple_tokenize(text: str) -> list[str]:
    """Lowercase tokenization tuned for BM25 and hash-based fallbacks."""
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def lexical_overlap_score(query: str, text: str) -> float:
    """A lightweight similarity fallback for reranking when no model is available."""
    query_counts = Counter(simple_tokenize(query))
    text_counts = Counter(simple_tokenize(text))
    if not query_counts or not text_counts:
        return 0.0

    overlap = sum(min(count, text_counts[token]) for token, count in query_counts.items())
    return overlap / max(sum(query_counts.values()), 1)


def stable_hash_embedding(text: str, dim: int = 256) -> list[float]:
    """Create a deterministic dense-like embedding without external models."""
    vector = [0.0] * dim
    tokens = simple_tokenize(text)
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[index] += sign

    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def mean(values: Iterable[float]) -> float:
    """Safe average helper."""
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def parse_datetime(value: str | None) -> datetime | None:
    """Best-effort ISO-ish datetime parsing for recency boosts."""
    if not value:
        return None

    cleaned = value.strip().replace("Z", "+00:00")
    for candidate in (cleaned, cleaned.split("T")[0]):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    return None
