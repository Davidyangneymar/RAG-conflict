"""Chunking utilities for retrieval-ready evidence units."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import RetrievalConfig
from src.schemas.documents import ChunkRecord, DocumentRecord
from src.utils.text import normalize_whitespace, simple_tokenize

ABBREVIATION_PATTERNS = (
    "U.S.",
    "U.K.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "St.",
    "vs.",
    "etc.",
)
ABBREVIATION_PLACEHOLDER = "\x00"
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(-])")


@dataclass
class SentenceUnit:
    """One sentence-like unit with offsets and grouping metadata."""

    text: str
    char_start: int
    char_end: int
    paragraph_index: int
    sentence_index: int

    @property
    def token_count(self) -> int:
        return len(simple_tokenize(self.text))


class TextSplitter:
    """Configurable splitter with a legacy token-window path and a v2 sentence-aware path."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        *,
        chunking_version: str = "v1",
        sentence_aware_chunking: bool = False,
        min_chunk_tokens: int = 24,
        target_chunk_tokens: int = 96,
        max_chunk_tokens: int = 144,
        sentence_overlap: int = 1,
        min_sentences_per_chunk: int = 2,
        filter_fragmentary_chunks: bool = True,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_version = chunking_version.strip().lower()
        self.sentence_aware_chunking = sentence_aware_chunking
        self.min_chunk_tokens = min_chunk_tokens
        self.target_chunk_tokens = target_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.sentence_overlap = sentence_overlap
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.filter_fragmentary_chunks = filter_fragmentary_chunks

    @classmethod
    def from_config(cls, config: RetrievalConfig) -> "TextSplitter":
        """Build a splitter from central configuration."""
        return cls(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            chunking_version=config.chunking_version,
            sentence_aware_chunking=config.sentence_aware_chunking,
            min_chunk_tokens=config.min_chunk_tokens,
            target_chunk_tokens=config.target_chunk_tokens,
            max_chunk_tokens=config.max_chunk_tokens,
            sentence_overlap=config.sentence_overlap,
            min_sentences_per_chunk=config.min_sentences_per_chunk,
            filter_fragmentary_chunks=config.filter_fragmentary_chunks,
        )

    def split_document(self, document: DocumentRecord) -> list[ChunkRecord]:
        """Split a normalized document into citation-friendly chunks."""
        use_v2 = self.sentence_aware_chunking or self.chunking_version == "v2"
        if use_v2:
            return self._split_document_v2(document)
        return self._split_document_v1(document)

    def _split_document_v1(self, document: DocumentRecord) -> list[ChunkRecord]:
        token_matches = list(re.finditer(r"\S+", document.full_text))
        if not token_matches:
            return []

        step = max(self.chunk_size - self.chunk_overlap, 1)
        chunks: list[ChunkRecord] = []

        for chunk_index, start_token_index in enumerate(range(0, len(token_matches), step)):
            end_token_index = min(start_token_index + self.chunk_size, len(token_matches))
            start_char = token_matches[start_token_index].start()
            end_char = token_matches[end_token_index - 1].end()
            text = normalize_whitespace(document.full_text[start_char:end_char])
            if not text:
                continue

            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.doc_id}::chunk-{chunk_index}",
                    doc_id=document.doc_id,
                    dataset=document.dataset,
                    source_name=document.source_name,
                    source_url=document.source_url,
                    title=document.title,
                    published_at=document.published_at,
                    text=text,
                    chunk_index=chunk_index,
                    char_start=start_char,
                    char_end=end_char,
                    metadata={
                        **document.metadata,
                        "language": document.language,
                    },
                )
            )

            if end_token_index == len(token_matches):
                break

        return chunks

    def _split_document_v2(self, document: DocumentRecord) -> list[ChunkRecord]:
        sentences = self._extract_sentence_units(document.full_text)
        if not sentences:
            return []

        groups = self._group_sentences(sentences)
        filtered_groups = self._filter_groups(groups)
        if not filtered_groups:
            filtered_groups = groups

        chunks: list[ChunkRecord] = []
        for chunk_index, group in enumerate(filtered_groups):
            start_char = group[0].char_start
            end_char = group[-1].char_end
            text = normalize_whitespace(document.full_text[start_char:end_char])
            if not text:
                continue

            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.doc_id}::chunk-{chunk_index}",
                    doc_id=document.doc_id,
                    dataset=document.dataset,
                    source_name=document.source_name,
                    source_url=document.source_url,
                    title=document.title,
                    published_at=document.published_at,
                    text=text,
                    chunk_index=chunk_index,
                    char_start=start_char,
                    char_end=end_char,
                    metadata={
                        **document.metadata,
                        "language": document.language,
                        "paragraph_start_index": group[0].paragraph_index,
                        "paragraph_end_index": group[-1].paragraph_index,
                        "sentence_start_index": group[0].sentence_index,
                        "sentence_end_index": group[-1].sentence_index,
                        "chunking_version": "v2",
                    },
                )
            )

        return chunks

    def _extract_sentence_units(self, text: str) -> list[SentenceUnit]:
        paragraphs = self._extract_paragraphs(text)
        sentence_units: list[SentenceUnit] = []
        global_sentence_index = 0

        for paragraph_index, (paragraph_text, paragraph_start) in enumerate(paragraphs):
            protected = paragraph_text
            for pattern in ABBREVIATION_PATTERNS:
                protected = protected.replace(pattern, pattern.replace(".", ABBREVIATION_PLACEHOLDER))
            protected = re.sub(r"\b([A-Z])\.", rf"\1{ABBREVIATION_PLACEHOLDER}", protected)

            cursor = 0
            segments = SENTENCE_BOUNDARY_RE.split(protected)
            for segment in segments:
                if not segment.strip():
                    cursor += len(segment)
                    continue

                found_at = protected.find(segment, cursor)
                if found_at < 0:
                    found_at = cursor
                cursor = found_at + len(segment)

                restored = segment.replace(ABBREVIATION_PLACEHOLDER, ".")
                trimmed = restored.strip()
                if not trimmed:
                    continue

                leading_ws = len(restored) - len(restored.lstrip())
                trailing_ws = len(restored) - len(restored.rstrip())
                relative_start = found_at + leading_ws
                relative_end = found_at + len(restored) - trailing_ws

                sentence_units.append(
                    SentenceUnit(
                        text=trimmed,
                        char_start=paragraph_start + relative_start,
                        char_end=paragraph_start + relative_end,
                        paragraph_index=paragraph_index,
                        sentence_index=global_sentence_index,
                    )
                )
                global_sentence_index += 1

        return sentence_units

    def _extract_paragraphs(self, text: str) -> list[tuple[str, int]]:
        normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs: list[tuple[str, int]] = []
        cursor = 0

        for block in re.split(r"\n\s*\n+", normalized_text):
            if not block.strip():
                cursor += len(block)
                continue

            start = normalized_text.find(block, cursor)
            if start < 0:
                start = cursor
            paragraphs.append((block, start))
            cursor = start + len(block)

        if paragraphs:
            return paragraphs

        stripped = normalized_text.strip()
        return [(stripped, 0)] if stripped else []

    def _group_sentences(self, sentences: list[SentenceUnit]) -> list[list[SentenceUnit]]:
        groups: list[list[SentenceUnit]] = []
        start = 0
        total_sentences = len(sentences)

        while start < total_sentences:
            end = start
            token_total = 0

            while end < total_sentences:
                next_sentence = sentences[end]
                next_total = token_total + next_sentence.token_count
                sentence_count = end - start + 1

                if sentence_count == 1 and next_sentence.token_count >= self.max_chunk_tokens:
                    token_total = next_total
                    end += 1
                    break

                if next_total > self.max_chunk_tokens and sentence_count >= self.min_sentences_per_chunk:
                    break

                token_total = next_total
                end += 1

                if token_total >= self.target_chunk_tokens and sentence_count >= self.min_sentences_per_chunk:
                    break

            if end <= start:
                end = start + 1

            group = sentences[start:end]

            remaining = total_sentences - end
            if (
                groups
                and remaining > 0
                and remaining < self.min_sentences_per_chunk
                and token_total + sum(item.token_count for item in sentences[end:]) <= self.max_chunk_tokens
            ):
                group = sentences[start:total_sentences]
                end = total_sentences

            groups.append(group)
            if end >= total_sentences:
                break

            next_start = max(start + 1, end - self.sentence_overlap)
            start = next_start

        return groups

    def _filter_groups(self, groups: list[list[SentenceUnit]]) -> list[list[SentenceUnit]]:
        if not self.filter_fragmentary_chunks:
            return groups

        filtered: list[list[SentenceUnit]] = []
        for group in groups:
            text = normalize_whitespace(" ".join(sentence.text for sentence in group))
            tokens = len(simple_tokenize(text))
            sentence_count = len(group)
            fragment_like = self._is_fragment_like(text)

            if tokens < self.min_chunk_tokens and filtered:
                filtered[-1].extend(group)
                continue

            if fragment_like and filtered:
                filtered[-1].extend(group)
                continue

            filtered.append(group)

        return filtered

    def _is_fragment_like(self, text: str) -> bool:
        tokens = simple_tokenize(text)
        if not tokens:
            return True
        if len(tokens) < self.min_chunk_tokens // 2:
            return True
        if text[0].islower() or not text[0].isalnum():
            return True
        if "-LRB-" in text or "-RRB-" in text or "-LSB-" in text or "-RSB-" in text:
            return True
        has_terminal_punct = any(text.endswith(marker) for marker in (".", "!", "?"))
        if len(tokens) <= 6 and not has_terminal_punct:
            return True
        alpha_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
        return len(alpha_tokens) <= 2
