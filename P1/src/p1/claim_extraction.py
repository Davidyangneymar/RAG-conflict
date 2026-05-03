from __future__ import annotations

import json
import logging
import os
import re
import socket
import ssl
import time
from hashlib import sha256
from urllib import error, request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from p1.observability import elapsed_ms, log_event
from p1.schemas import ChunkInput, Claim, ClaimSource


logger = logging.getLogger(__name__)


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9_-]{1,}\b")
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
ABBREVIATION_PLACEHOLDER = "<DOT>"
QUOTE_SPAN_RE = re.compile(r"'[^']{2,}'|\"[^\"]{2,}\"|“[^”]{2,}”|‘[^’]{2,}’")
TIME_RE = re.compile(
    r"\b(?:"
    r"\d{4}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:,\s+\d{4})?"
    r"|(?:yesterday|today|tomorrow|last year|next year|this year)"
    r")\b",
    re.IGNORECASE,
)
ENTITY_STOPWORDS = {"A", "An", "The", "This", "That", "These", "Those"}
NEGATIVE_HINTS = {"not", "no", "never", "false", "fake", "denied", "deny", "without", "hoax"}
UNCERTAIN_HINTS = {"may", "might", "could", "reportedly", "allegedly", "unclear", "possible", "possibly"}
RELATION_HINTS = (
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "find",
    "finds",
    "found",
    "flee",
    "flees",
    "led",
    "lead",
    "leads",
    "ignored",
    "ignore",
    "ignores",
    "detained",
    "detain",
    "detains",
    "reported",
    "claims",
    "claimed",
    "claim",
    "deny",
    "denies",
    "denied",
    "says",
    "said",
    "reveals",
    "revealed",
    "finds",
    "killed",
    "injured",
    "arrested",
    "open",
    "opens",
    "opened",
    "pass",
    "passes",
    "follows",
    "followed",
    "cited",
    "survives",
    "survive",
    "landed",
    "lands",
    "launch",
    "launches",
    "launching",
    "talk",
    "talks",
    "burrow",
    "burrows",
    "burrowed",
    "make",
    "makes",
    "made",
    "rip",
    "rips",
    "ripped",
    "exit",
    "exits",
    "tried",
    "try",
    "keep",
    "keeps",
    "bail",
    "bailed",
    "post",
    "posts",
    "want",
    "wants",
    "question",
    "questions",
    "questioned",
    "decrease",
    "decreases",
    "decreased",
    "identified",
    "identify",
    "identifies",
    "contaminated",
    "infect",
    "infected",
    "needed",
    "need",
    "needs",
    "assassinated",
    "set",
    "sets",
    "agree",
    "agrees",
    "agreed",
    "posted",
    "debunked",
    "stolen",
    "trigger",
    "triggers",
    "emerge",
    "emerges",
    "emerged",
    "sack",
    "sacked",
    "caught",
    "behead",
    "beheads",
    "beheaded",
    "add",
    "adds",
    "added",
    "reduces",
    "reduced",
    "confirms",
    "confirmed",
)
REPORTING_RELATIONS = {
    "claim",
    "claims",
    "claimed",
    "reported",
    "says",
    "said",
    "reveals",
    "revealed",
    "confirms",
    "confirmed",
}
MONTH_TOKENS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
QUANTITY_ENTITY_HINTS = {"hundreds", "thousands", "millions", "billions"}
SUBJECT_PREFIX_NOISE = {"no", "that", "this", "these", "those", "didn't", "didnt", "may", "soon"}
SUBJECT_SUFFIX_NOISE = {
    "reportedly",
    "allegedly",
    "severely",
    "same",
    "real",
    "identity",
    "report",
    "didn",
    "t",
    "just",
    "out",
    "accidentally",
}
TRAILING_PREPOSITIONS = {"in", "on", "at", "for", "to", "of", "by", "with", "from", "as"}
QUESTION_PREFIXES = {"who", "what", "when", "where", "why", "how", "would", "could", "should", "is", "are", "did", "do", "does", "can", "will"}
LLM_PROMPT_VARIANTS = {"baseline", "headline_aware"}


class ClaimExtractor(Protocol):
    def extract(self, chunk: ChunkInput) -> list[Claim]:
        ...

    def extract_many(self, chunks: list[ChunkInput]) -> list[Claim]:
        ...


@dataclass
class SentenceClaimExtractor:
    min_chars: int = 15
    entity_backend: str = "auto"

    def extract(self, chunk: ChunkInput) -> list[Claim]:
        sentences = _split_sentences(chunk.text)
        claims: list[Claim] = []
        allow_question_claims = str(chunk.metadata.get("role") or "").strip().lower() == "query"

        for index, sentence in enumerate(sentences):
            if len(sentence) < self.min_chars:
                continue
            if _is_question_like(sentence) and not allow_question_claims:
                continue

            claim_id = f"{chunk.doc_id}:{chunk.chunk_id or 'chunk0'}:sent{index}"
            entities = extract_entity_candidates(sentence, backend=self.entity_backend)
            normalized_time = _extract_time(sentence)
            polarity = _infer_polarity(sentence)
            certainty = _infer_certainty(sentence)
            claims.append(
                Claim(
                    claim_id=claim_id,
                    text=sentence,
                    source=ClaimSource(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        sentence_id=str(index),
                        metadata=chunk.metadata,
                    ),
                    source_doc_id=chunk.doc_id,
                    source_chunk_id=chunk.chunk_id,
                    entities=entities,
                    subject=entities[0] if entities else None,
                    time=normalized_time,
                    polarity=polarity,
                    certainty=certainty,
                    metadata={"baseline": "sentence_split", "time_extracted": normalized_time is not None},
                )
            )
        return claims

    def extract_many(self, chunks: list[ChunkInput]) -> list[Claim]:
        """Extract sentence-level baseline claims from multiple chunks."""
        started_at = time.perf_counter()
        claims: list[Claim] = []
        for chunk in chunks:
            claims.extend(self.extract(chunk))
        log_event(
            logger,
            "p1.claim_extraction.sentence.complete",
            level=logging.DEBUG,
            chunks=len(chunks),
            claims=len(claims),
            duration_ms=elapsed_ms(started_at, time.perf_counter()),
        )
        return claims


@dataclass
class StructuredClaimExtractor:
    min_chars: int = 15
    entity_backend: str = "auto"

    def extract(self, chunk: ChunkInput) -> list[Claim]:
        sentence_extractor = SentenceClaimExtractor(min_chars=self.min_chars, entity_backend=self.entity_backend)
        sentence_claims = sentence_extractor.extract(chunk)
        structured_claims: list[Claim] = []

        for claim in sentence_claims:
            parsed = _parse_structured_fields(claim.text, claim.entities, claim.time)
            structured_claims.append(
                Claim(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    source=claim.source,
                    source_doc_id=claim.source_doc_id,
                    source_chunk_id=claim.source_chunk_id,
                    entities=claim.entities,
                    subject=parsed["subject"],
                    relation=parsed["relation"],
                    object=parsed["object"],
                    qualifier=parsed["qualifier"],
                    time=claim.time,
                    polarity=claim.polarity,
                    certainty=claim.certainty,
                    metadata={
                        **claim.metadata,
                        "baseline": "structured_heuristic",
                        "structured_fields_present": _has_usable_structure(parsed),
                        "full_triplet_present": bool(parsed["subject"] and parsed["relation"] and parsed["object"]),
                    },
                )
            )
        return structured_claims

    def extract_many(self, chunks: list[ChunkInput]) -> list[Claim]:
        """Extract heuristic subject-relation-object claims from multiple chunks."""
        started_at = time.perf_counter()
        structured_claims: list[Claim] = []
        for chunk in chunks:
            structured_claims.extend(self.extract(chunk))
        log_event(
            logger,
            "p1.claim_extraction.structured.complete",
            level=logging.DEBUG,
            chunks=len(chunks),
            claims=len(structured_claims),
            duration_ms=elapsed_ms(started_at, time.perf_counter()),
        )
        return structured_claims


@dataclass
class LLMClaimExtractor:
    min_chars: int = 15
    entity_backend: str = "auto"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    api_style: str = "chat_completions"
    prompt_variant: str = "baseline"
    timeout_seconds: float = 45.0
    temperature: float = 0.0
    fallback_to_heuristic: bool = False
    batch_size: int = 1

    def __post_init__(self) -> None:
        normalized_variant = self.prompt_variant.strip().lower()
        if normalized_variant not in LLM_PROMPT_VARIANTS:
            raise ValueError(f"Unsupported prompt variant: {self.prompt_variant}")
        self.prompt_variant = normalized_variant

    def extract(self, chunk: ChunkInput) -> list[Claim]:
        return self.extract_many([chunk])

    def extract_many(self, chunks: list[ChunkInput]) -> list[Claim]:
        """Extract structured claims with an LLM, using cache and heuristic fallback."""
        started_at = time.perf_counter()
        sentence_extractor = SentenceClaimExtractor(min_chars=self.min_chars, entity_backend=self.entity_backend)
        sentence_claims: list[tuple[Claim, ChunkInput]] = []
        completed_claims: dict[str, Claim] = {}
        cache_hits = 0
        runtime_fallbacks = 0
        request_failures = 0
        batch_count = 0
        for chunk in chunks:
            for claim in sentence_extractor.extract(chunk):
                sentence_claims.append((claim, chunk))

        runtime = self._resolve_runtime()
        pending_items: list[dict[str, object]] = []

        for claim, chunk in sentence_claims:
            if runtime["error"] is not None:
                runtime_fallbacks += 1
                completed_claims[claim.claim_id] = self._build_structured_claim(
                    claim,
                    parsed=None,
                    metadata=self._default_llm_metadata(runtime, error=str(runtime["error"])),
                )
                continue

            metadata = self._default_llm_metadata(runtime)
            cache_key = _build_llm_cache_key(
                model=str(runtime["model"]),
                base_url=str(runtime["base_url"]),
                api_style=str(runtime["api_style"]),
                prompt_variant=self.prompt_variant,
                text=claim.text,
                dataset=str(chunk.metadata.get("dataset") or ""),
                role=str(chunk.metadata.get("role") or ""),
            )
            cached_payload = _read_llm_cache(cache_key)
            if cached_payload is not None:
                cache_hits += 1
                metadata["llm_used"] = True
                metadata["llm_cache_hit"] = True
                completed_claims[claim.claim_id] = self._build_structured_claim(
                    claim,
                    parsed=cached_payload,
                    metadata=metadata,
                )
                continue

            pending_items.append(
                {
                    "item_id": claim.claim_id,
                    "claim": claim,
                    "chunk": chunk,
                    "cache_key": cache_key,
                    "metadata": metadata,
                }
            )

        log_event(
            logger,
            "p1.claim_extraction.llm.prepare",
            chunks=len(chunks),
            sentence_claims=len(sentence_claims),
            cache_hits=cache_hits,
            pending=len(pending_items),
            runtime_error=runtime["error"],
            prompt_variant=self.prompt_variant,
            api_style=runtime["api_style"],
        )

        if runtime["error"] is None and pending_items:
            effective_batch_size = max(1, self.batch_size, _read_int_env("P1_LLM_BATCH_SIZE", self.batch_size))
            for start in range(0, len(pending_items), effective_batch_size):
                batch = pending_items[start : start + effective_batch_size]
                batch_count += 1
                try:
                    batch_payload = self._extract_structured_fields_batch(batch, runtime)
                except RuntimeError as exc:
                    request_failures += len(batch)
                    batch_payload = {str(item["item_id"]): None for item in batch}
                    for item in batch:
                        item_metadata = dict(item["metadata"])
                        item_metadata["llm_error"] = str(exc)
                        item["metadata"] = item_metadata

                for item in batch:
                    claim = item["claim"]
                    cache_key = str(item["cache_key"])
                    item_metadata = dict(item["metadata"])
                    item_metadata["llm_batch_size"] = len(batch)
                    parsed = batch_payload.get(str(item["item_id"]))
                    if parsed is not None:
                        item_metadata["llm_used"] = True
                        item_metadata["llm_cache_hit"] = False
                        _write_llm_cache(cache_key, parsed)
                    completed_claims[claim.claim_id] = self._build_structured_claim(
                        claim,
                        parsed=parsed,
                        metadata=item_metadata,
                    )
        log_event(
            logger,
            "p1.claim_extraction.llm.complete",
            claims=len(completed_claims),
            cache_hits=cache_hits,
            requested=len(pending_items),
            batches=batch_count,
            runtime_fallbacks=runtime_fallbacks,
            request_failures=request_failures,
            duration_ms=elapsed_ms(started_at, time.perf_counter()),
        )
        return [completed_claims[claim.claim_id] for claim, _ in sentence_claims]

    def _default_llm_metadata(
        self,
        runtime: dict[str, str | None],
        *,
        error: str | None = None,
    ) -> dict[str, str | bool | int | None]:
        return {
            "prompt_variant": self.prompt_variant,
            "llm_model": runtime["model"],
            "llm_api_style": runtime["api_style"],
            "llm_used": False,
            "llm_fallback_used": False,
            "llm_error": error,
        }

    def _resolve_runtime(self) -> dict[str, str | None]:
        api_key = self.api_key or os.getenv("P1_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        model = self.model or os.getenv("P1_LLM_MODEL")
        base_url = self.base_url or os.getenv("P1_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        api_style = (os.getenv("P1_LLM_API_STYLE") or self.api_style or "chat_completions").strip().lower()
        if not api_key:
            return {"api_key": None, "model": model, "base_url": base_url, "api_style": api_style, "error": "missing_api_key"}
        if not model:
            return {"api_key": api_key, "model": None, "base_url": base_url, "api_style": api_style, "error": "missing_model"}
        return {"api_key": api_key, "model": model, "base_url": base_url, "api_style": api_style, "error": None}

    def _extract_structured_fields_batch(
        self,
        batch: list[dict[str, object]],
        runtime: dict[str, str | None],
    ) -> dict[str, dict[str, str | None] | None]:
        messages = _build_llm_batch_messages(
            items=[
                {
                    "item_id": str(item["item_id"]),
                    "text": str(item["claim"].text),
                    "dataset": str(item["chunk"].metadata.get("dataset") or ""),
                    "role": str(item["chunk"].metadata.get("role") or ""),
                }
                for item in batch
            ],
            prompt_variant=self.prompt_variant,
        )
        if runtime["api_style"] == "responses":
            response_json = _post_responses_api(
                api_key=str(runtime["api_key"]),
                base_url=str(runtime["base_url"]),
                model=str(runtime["model"]),
                messages=messages,
                timeout_seconds=self.timeout_seconds,
                temperature=self.temperature,
            )
            return _extract_batch_json_from_responses_api(response_json)
        response_json = _post_chat_completion(
            api_key=str(runtime["api_key"]),
            base_url=str(runtime["base_url"]),
            model=str(runtime["model"]),
            messages=messages,
            timeout_seconds=self.timeout_seconds,
            temperature=self.temperature,
        )
        return _extract_batch_json_from_chat_completion(response_json)

    def _build_structured_claim(
        self,
        claim: Claim,
        *,
        parsed: dict[str, str | None] | None,
        metadata: dict[str, str | bool | int | None],
    ) -> Claim:
        if parsed is None:
            if self.fallback_to_heuristic:
                parsed = _parse_structured_fields(claim.text, claim.entities, claim.time)
                metadata = {
                    **metadata,
                    "llm_fallback_used": True,
                    "fallback_backend": "structured_heuristic",
                }
            else:
                parsed = {"subject": None, "relation": None, "object": None, "qualifier": None, "time": None}
        normalized = _normalize_llm_structured_fields(parsed, fallback_time=claim.time)
        return Claim(
            claim_id=claim.claim_id,
            text=claim.text,
            source=claim.source,
            source_doc_id=claim.source_doc_id,
            source_chunk_id=claim.source_chunk_id,
            entities=claim.entities,
            subject=normalized["subject"],
            relation=normalized["relation"],
            object=normalized["object"],
            qualifier=normalized["qualifier"],
            time=normalized["time"],
            polarity=claim.polarity,
            certainty=claim.certainty,
            metadata={
                **claim.metadata,
                **metadata,
                "baseline": "structured_llm",
                "structured_fields_present": _has_usable_structure(normalized),
                "full_triplet_present": bool(
                    normalized["subject"] and normalized["relation"] and normalized["object"]
                ),
            },
        )


def build_claim_extractor(
    kind: str = "sentence",
    entity_backend: str = "auto",
    *,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
    llm_api_style: str = "chat_completions",
    prompt_variant: str = "baseline",
    llm_timeout_seconds: float = 45.0,
    llm_temperature: float = 0.0,
    fallback_to_heuristic: bool = False,
    llm_batch_size: int = 1,
) -> ClaimExtractor:
    normalized = kind.strip().lower()
    if normalized == "sentence":
        return SentenceClaimExtractor(entity_backend=entity_backend)
    if normalized in {"structured", "structured_heuristic"}:
        return StructuredClaimExtractor(entity_backend=entity_backend)
    if normalized in {"llm", "structured_llm"}:
        return LLMClaimExtractor(
            entity_backend=entity_backend,
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url,
            api_style=llm_api_style,
            prompt_variant=prompt_variant,
            timeout_seconds=llm_timeout_seconds,
            temperature=llm_temperature,
            fallback_to_heuristic=fallback_to_heuristic,
            batch_size=llm_batch_size,
        )
    raise ValueError(f"Unsupported claim extractor kind: {kind}")


def extract_entity_candidates(text: str, backend: str = "auto") -> list[str]:
    normalized = backend.strip().lower()
    if normalized in {"auto", "spacy"}:
        entities = _extract_with_spacy(text)
        if entities:
            return entities
        if normalized == "spacy":
            return []
    return sorted({item for item in ENTITY_RE.findall(text) if item not in ENTITY_STOPWORDS})


@lru_cache(maxsize=1)
def _load_spacy_model():
    try:
        import spacy
    except ImportError:
        return None

    candidate_paths = [
        Path("manual_models/en_core_web_sm"),
        Path("manual_models/spacy/en_core_web_sm"),
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            try:
                return spacy.load(str(candidate.resolve()))
            except OSError:
                continue

    for model_name in ("en_core_web_sm",):
        try:
            return spacy.load(model_name)
        except OSError:
            continue
    return None


def _extract_with_spacy(text: str) -> list[str]:
    nlp = _load_spacy_model()
    if nlp is None:
        return []
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        cleaned = ent.text.strip()
        if cleaned and cleaned not in ENTITY_STOPWORDS:
            entities.append(cleaned)
    return sorted(set(entities))


def _extract_time(text: str) -> str | None:
    match = TIME_RE.search(text)
    if not match:
        return None
    return match.group(0)


def _infer_polarity(text: str) -> str:
    lowered_tokens = {token.lower() for token in re.findall(r"\b[a-zA-Z]+\b", text)}
    if lowered_tokens & NEGATIVE_HINTS:
        return "negative"
    if lowered_tokens & UNCERTAIN_HINTS:
        return "uncertain"
    return "positive"


def _infer_certainty(text: str) -> float:
    lowered_tokens = {token.lower() for token in re.findall(r"\b[a-zA-Z]+\b", text)}
    if lowered_tokens & UNCERTAIN_HINTS:
        return 0.45
    if lowered_tokens & NEGATIVE_HINTS:
        return 0.7
    return 0.85


def _is_question_like(text: str) -> bool:
    stripped = text.strip()
    tokens = re.findall(r"[A-Za-z]+", stripped.lower())
    if not tokens:
        return False
    if stripped.endswith("?"):
        return tokens[0] in QUESTION_PREFIXES
    return tokens[0] == "why"


def _has_usable_structure(parsed: dict[str, str | None]) -> bool:
    slot_count = sum(1 for value in parsed.values() if value)
    return bool(parsed.get("relation")) and slot_count >= 2


def _parse_structured_fields(text: str, entities: list[str], time_value: str | None = None) -> dict[str, str | None]:
    normalized_text = re.sub(r"\s+", " ", text).strip()
    lowered = normalized_text.lower()
    content_entities = []
    for entity in entities:
        normalized_entity = _normalize_slot_entity(entity)
        if _is_content_entity(normalized_entity):
            content_entities.append(normalized_entity)

    relation, relation_match = _find_relation(lowered)
    subject = _pick_subject_entity(normalized_text, content_entities, relation_match)
    if subject is None:
        subject = _pick_subject_from_prefix(normalized_text, relation_match)

    object_value = None
    qualifier = None
    if relation_match is not None:
        if relation == "is" and lowered[relation_match.start() : relation_match.end()] == "not dead":
            object_value = normalized_text[relation_match.start() : relation_match.end()].strip(" ,:-")
        else:
            object_value = normalized_text[relation_match.end() :].strip(" ,:-")
        if subject and object_value.lower().startswith(subject.lower()):
            object_value = object_value[len(subject) :].strip(" ,:-")
    elif content_entities:
        trailing_entities = [item for item in content_entities[1:] if item != subject]
        if trailing_entities:
            object_value = ", ".join(trailing_entities[:2])

    if ":" in normalized_text:
        qualifier = normalized_text.split(":", 1)[0].strip()
    elif "," in normalized_text and len(normalized_text.split(",", 1)[0].split()) <= 6:
        qualifier = normalized_text.split(",", 1)[0].strip()

    if object_value is not None:
        object_value = _clean_object_text(object_value, relation, time_value)
    if qualifier:
        qualifier = _clean_qualifier_text(qualifier)

    return {
        "subject": subject,
        "relation": relation,
        "object": object_value,
        "qualifier": qualifier,
    }


def _split_sentences(text: str) -> list[str]:
    protected = text
    for pattern in ABBREVIATION_PATTERNS:
        protected = protected.replace(pattern, pattern.replace(".", ABBREVIATION_PLACEHOLDER))
    sentences = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(protected) if segment.strip()]
    return [segment.replace(ABBREVIATION_PLACEHOLDER, ".") for segment in sentences]


def _find_relation(lowered_text: str) -> tuple[str | None, re.Match[str] | None]:
    quote_spans = [match.span() for match in QUOTE_SPAN_RE.finditer(lowered_text)]
    candidates: list[tuple[int, int, int, str, re.Match[str]]] = []
    for hint in RELATION_HINTS:
        match = re.search(rf"\b{re.escape(hint)}\b", lowered_text)
        if match:
            in_quote = any(start <= match.start() < end for start, end in quote_spans)
            candidates.append((1 if in_quote else 0, match.start(), -(match.end() - match.start()), hint, match))

    contraction_patterns = (
        (r"\bit[’']s\b", "is"),
        (r"\bisn[’']t\b", "is"),
        (r"\baren[’']t\b", "are"),
        (r"\bwasn[’']t\b", "was"),
        (r"\bweren[’']t\b", "were"),
        (r"\bhasn[’']t\b", "has"),
        (r"\bhaven[’']t\b", "have"),
    )
    for pattern, relation in contraction_patterns:
        contraction_match = re.search(pattern, lowered_text)
        if contraction_match:
            candidates.append((0, contraction_match.start(), -(contraction_match.end() - contraction_match.start()), relation, contraction_match))
    not_dead_match = re.search(r"\bnot dead\b", lowered_text)
    if not_dead_match:
        candidates.append((0, not_dead_match.start(), -(not_dead_match.end() - not_dead_match.start()), "is", not_dead_match))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, _, relation, match = candidates[0]
    return relation, match


def _pick_subject_entity(
    text: str,
    entities: list[str],
    relation_match: re.Match[str] | None,
) -> str | None:
    if relation_match is None:
        positions: list[tuple[int, int, str]] = []
        lowered_text = text.lower()
        for entity in entities:
            position = lowered_text.find(entity.lower())
            if position >= 0:
                positions.append((position, len(entity), entity))
        if positions:
            positions.sort(key=lambda item: (item[0], -item[1]))
            return _clean_subject_text(positions[0][2])
        return _pick_subject_from_prefix(_strip_quoted_segments(text), None) or (_clean_subject_text(entities[0]) if entities else None)

    relation_start = relation_match.start()
    entity_candidates: list[tuple[int, int, str]] = []
    lowered_text = text.lower()
    for entity in entities:
        if TIME_RE.fullmatch(entity):
            continue
        position = lowered_text.find(entity.lower())
        if 0 <= position < relation_start:
            entity_candidates.append((position, len(entity), entity))
    if entity_candidates:
        entity_candidates.sort(key=lambda item: (item[0], item[1]))
        chosen = _clean_subject_text(entity_candidates[-1][2])
        if not chosen:
            return _pick_subject_from_prefix(
                _strip_quoted_segments(text[:relation_start]),
                relation_match,
            )
        raw_prefix = text[:relation_start]
        prefix_candidate = _pick_titlecase_prefix_subject(_strip_quoted_segments(raw_prefix))
        if prefix_candidate and len(chosen.split()) == 1:
            return prefix_candidate
        if len(chosen.split()) == 1:
            leading_candidate = _pick_leading_subject_token(_strip_quoted_segments(raw_prefix))
            if leading_candidate and leading_candidate.lower() != chosen.lower():
                return leading_candidate
        return chosen

    prefix = _strip_quoted_segments(text[:relation_start]).strip(" ,:-")
    if ":" in prefix:
        prefix = prefix.split(":", 1)[-1].strip()
    return _clean_subject_text(_pick_subject_from_prefix(prefix, None))


def _pick_subject_from_prefix(text: str, relation_match: re.Match[str] | None) -> str | None:
    if relation_match is None:
        prefix = text.strip(" ,:-")
    else:
        prefix = text[: relation_match.start()].strip(" ,:-")
    prefix = _strip_quoted_segments(prefix)
    if ":" in prefix:
        prefix = prefix.split(":", 1)[-1].strip()
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", prefix)
    if not tokens:
        return None
    while tokens and tokens[0].lower() in SUBJECT_PREFIX_NOISE:
        tokens = tokens[1:]
    while tokens and tokens[-1].lower() in SUBJECT_PREFIX_NOISE | TRAILING_PREPOSITIONS:
        tokens = tokens[:-1]
    if not tokens:
        return None
    if len(tokens) > 5:
        tokens = tokens[:5]
    candidate = " ".join(tokens).strip()
    return _clean_subject_text(candidate)


def _is_content_entity(entity: str) -> bool:
    if not entity:
        return False
    if TIME_RE.fullmatch(entity):
        return False
    stripped = entity.strip()
    if not stripped or not stripped[0].isalpha() or stripped[0].islower():
        return False
    if not re.search(r"[A-Za-z]", entity):
        return False
    lowered = stripped.lower()
    if lowered in MONTH_TOKENS or lowered in QUANTITY_ENTITY_HINTS:
        return False
    if lowered.startswith("at least "):
        return False
    return True


def _normalize_slot_entity(entity: str) -> str:
    cleaned = entity.strip()
    cleaned = cleaned.strip(" '\"“”‘’.,:;!?")
    if "'" in cleaned or '"' in cleaned or "“" in cleaned or "‘" in cleaned:
        segments = [segment.strip(" '\"“”‘’.,:;!?") for segment in re.split(r"[\"'“”‘’]+", cleaned) if segment.strip(" '\"“”‘’.,:;!?")]
        if segments:
            cleaned = segments[0]
    cleaned = re.sub(r"\s+", " ", cleaned)
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", cleaned)
    lowered_tokens = [token.lower() for token in tokens]
    for index, token in enumerate(lowered_tokens):
        if token in RELATION_HINTS and index > 0:
            cleaned = " ".join(tokens[:index])
            break
    if len(cleaned.split()) > 6:
        cleaned = " ".join(cleaned.split()[:6])
    return cleaned


def _strip_quoted_segments(text: str) -> str:
    stripped = re.sub(r"'[^']{2,}'|\"[^\"]{2,}\"|“[^”]{2,}”|‘[^’]{2,}’", " ", text)
    return re.sub(r"\s+", " ", stripped).strip()


def _clean_subject_text(subject: str | None) -> str | None:
    if not subject:
        return None
    cleaned = subject.strip(" ,:-'\"“”‘’.")
    cleaned = _strip_quoted_segments(cleaned)
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", cleaned)
    while tokens and tokens[0].lower() in SUBJECT_PREFIX_NOISE:
        tokens = tokens[1:]
    while tokens and tokens[-1].lower() in SUBJECT_SUFFIX_NOISE | SUBJECT_PREFIX_NOISE | TRAILING_PREPOSITIONS:
        tokens = tokens[:-1]
    if not tokens:
        return None
    if len(tokens) > 5:
        tokens = tokens[-5:]
    candidate = " ".join(tokens).strip()
    return candidate or None


def _pick_titlecase_prefix_subject(prefix: str) -> str | None:
    if "'" in prefix or '"' in prefix or "“" in prefix or "‘" in prefix:
        return None
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", prefix)
    if not words:
        return None
    groups: list[list[str]] = []
    current: list[str] = []
    for word in words:
        if word[0].isupper():
            current.append(word)
        else:
            if current:
                groups.append(current)
                current = []
    if current:
        groups.append(current)
    if not groups:
        return None
    groups.sort(key=len, reverse=True)
    chosen = groups[0]
    if len(chosen) < 2:
        return None
    return " ".join(chosen[:4]).strip()


def _pick_leading_subject_token(prefix: str) -> str | None:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", prefix)
    if not words:
        return None
    first = words[0].strip()
    return first if first and first[0].isupper() else None


def _clean_object_text(object_text: str, relation: str | None, time_value: str | None) -> str | None:
    cleaned = object_text.strip(" ,:-")
    cleaned = re.sub(r"\b[’']t\b", " not", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(that|the)\s+", "", cleaned, flags=re.IGNORECASE)
    if relation in {"pass", "passes", "rip", "rips", "ripped"}:
        cleaned = re.sub(r"^(on|up)\s+", "", cleaned, flags=re.IGNORECASE)
    if relation in {"talk", "talks"}:
        cleaned = re.sub(r"^for\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.split(r"\blaunching\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,:-")
    if relation in {"tried", "try"}:
        cleaned = re.sub(r"^out\s+", "", cleaned, flags=re.IGNORECASE)
    if relation in REPORTING_RELATIONS:
        cleaned = re.sub(r"^(reportedly|allegedly|reportedly\s+the|allegedly\s+the)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.split(r"[\"“”]|'\s+(?=[A-Z])", cleaned, maxsplit=1)[0].strip(" ,:-")
    cleaned = re.split(r"\s+\(", cleaned, maxsplit=1)[0].strip(" ,:-")
    if time_value:
        cleaned = re.sub(rf"([,\s]+{re.escape(time_value)})$", "", cleaned, flags=re.IGNORECASE).strip(" ,")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;-")
    if relation in REPORTING_RELATIONS:
        embedded_relation, embedded_match = _find_relation(cleaned.lower())
        if embedded_relation and embedded_match and embedded_match.start() <= 20:
            cleaned = cleaned[embedded_match.start() :].strip(" ,:-")
    if relation in {"says", "said"}:
        embedded_reporting = re.search(r"\b(says|said|claims|claimed)\b", cleaned, flags=re.IGNORECASE)
        if embedded_reporting and embedded_reporting.start() >= 8:
            cleaned = cleaned[embedded_reporting.end() :].strip(" ,:-")
    if relation in {"keep", "keeps"}:
        cleaned = re.split(r"\bthey\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,:-")
    if relation in {"says", "said", "claims", "claim", "claimed"}:
        cleaned = re.split(r"\b(that will|that would|that could)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,:-")
    if relation in {"says", "said"}:
        cleaned = re.split(r"[;:]", cleaned, maxsplit=1)[0].strip(" ,:-")
    if relation in {"needed", "need", "needs"}:
        cleaned = re.sub(r"^on\s+", "", cleaned, flags=re.IGNORECASE)
    if relation in {"post", "posts", "posted"}:
        cleaned = re.split(r"\bclaiming to\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,:-")
    if relation in {"lands", "landed"}:
        cleaned = re.split(r"\bwhich\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,:-")
    if relation in {"have"}:
        cleaned = re.split(r"[:;]", cleaned, maxsplit=1)[0].strip(" ,:-")
    if relation in {"behead", "beheads", "beheaded"}:
        cleaned = re.split(r"\bas warning\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,:-")
    if relation in {"debunked"} and not cleaned:
        cleaned = "false"
    cleaned = re.split(r",\s+(?=[A-Za-z])", cleaned, maxsplit=1)[0].strip(" ,:-")
    cleaned = re.split(r"\b(where|after|despite)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,:-")
    cleaned = re.sub(r"\b(Iraq says|actor reportedly felt.*|Fake News Story Goes Viral)$", "", cleaned, flags=re.IGNORECASE).strip(" ,:-")
    cleaned = re.sub(r"\b(from|in|on|at|by|of|for|to)$", "", cleaned, flags=re.IGNORECASE).strip(" ,:-")
    if len(cleaned.split()) > 12:
        cleaned = " ".join(cleaned.split()[:12])
    cleaned = cleaned.strip(" .,:;-?")
    return cleaned if cleaned else None


def _clean_qualifier_text(qualifier: str) -> str | None:
    cleaned = qualifier.strip(" ,:-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned.split()) > 6:
        cleaned = " ".join(cleaned.split()[:6])
    return cleaned[:80].strip() or None


def _clean_relation_text(relation: str | None) -> str | None:
    if not relation:
        return None
    cleaned = relation.strip().lower()
    cleaned = cleaned.strip(" .,:;!?\"'“”‘’")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = " ".join(cleaned.split()[:3])
    return cleaned or None


def _normalize_llm_structured_fields(
    parsed: dict[str, str | None],
    *,
    fallback_time: str | None,
) -> dict[str, str | None]:
    normalized_time = parsed.get("time") or fallback_time
    relation = _clean_relation_text(parsed.get("relation"))
    return {
        "subject": _clean_subject_text(parsed.get("subject")),
        "relation": relation,
        "object": _clean_object_text(parsed.get("object") or "", relation, normalized_time) if parsed.get("object") else None,
        "qualifier": _clean_qualifier_text(parsed.get("qualifier") or "") if parsed.get("qualifier") else None,
        "time": normalized_time.strip() if isinstance(normalized_time, str) and normalized_time.strip() else None,
    }


def _build_llm_messages(
    *,
    text: str,
    prompt_variant: str,
    dataset: str,
    role: str,
) -> list[dict[str, str]]:
    headline_guidance = ""
    if prompt_variant == "headline_aware":
        headline_guidance = (
            "Treat news headlines as compressed claims. Prefer the real actor or topic as subject, "
            "not a location or date token. If the text is about a hoax, denial, or 'not dead' style correction, "
            "use a short copular relation such as 'is' when appropriate.\n"
        )
    system_prompt = (
        "You extract one structured factual claim from a short text span.\n"
        "Return valid JSON only with keys: subject, relation, object, qualifier, time.\n"
        "Rules:\n"
        "- Keep values short and literal.\n"
        "- Use null when a field is absent.\n"
        "- relation should be a short predicate, ideally 1-3 words.\n"
        "- object should exclude trailing context, source attributions, and duplicated subject text.\n"
        "- qualifier is for headline prefix, source framing, or narrow context.\n"
        "- Do not include explanations or markdown.\n"
        f"{headline_guidance}"
    )
    user_prompt = (
        f"dataset={dataset or 'unknown'}\n"
        f"role={role or 'claim'}\n"
        "text:\n"
        f"{text}\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_llm_batch_messages(
    *,
    items: list[dict[str, str]],
    prompt_variant: str,
) -> list[dict[str, str]]:
    headline_guidance = ""
    if prompt_variant == "headline_aware":
        headline_guidance = (
            "Treat news headlines as compressed claims. Prefer the real actor or topic as subject, "
            "not a location or date token. If the text is about a hoax, denial, or 'not dead' style correction, "
            "use a short copular relation such as 'is' when appropriate.\n"
        )
    system_prompt = (
        "You extract one structured factual claim for each input item.\n"
        "Return valid JSON only in this exact shape:\n"
        '{"items":[{"item_id":"...","subject":null,"relation":null,"object":null,"qualifier":null,"time":null}]}\n'
        "Rules:\n"
        "- Return exactly one result per input item.\n"
        "- Preserve item_id exactly.\n"
        "- Keep values short and literal.\n"
        "- Use null when a field is absent.\n"
        "- relation should be a short predicate, ideally 1-3 words.\n"
        "- object should exclude trailing context, source attributions, and duplicated subject text.\n"
        "- qualifier is for headline prefix, source framing, or narrow context.\n"
        "- Do not include explanations or markdown.\n"
        f"{headline_guidance}"
    )
    user_prompt = "items:\n" + json.dumps(items, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _post_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
    temperature: float,
) -> dict:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    req = request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    return _execute_llm_request(req=req, timeout_seconds=timeout_seconds)


def _post_responses_api(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
    temperature: float,
) -> dict:
    endpoint = f"{base_url.rstrip('/')}/responses"
    payload = json.dumps(
        {
            "model": model,
            "stream": False,
            "temperature": temperature,
            "input": _build_responses_input(messages),
        }
    ).encode("utf-8")
    req = request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    return _execute_llm_request(req=req, timeout_seconds=timeout_seconds)


def _build_ssl_context() -> ssl.SSLContext:
    if os.getenv("P1_LLM_SKIP_SSL_VERIFY", "").strip().lower() in {"1", "true", "yes"}:
        return ssl._create_unverified_context()

    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()

    certifi_path = certifi.where()
    if certifi_path and Path(certifi_path).exists():
        try:
            return ssl.create_default_context(cafile=certifi_path)
        except OSError:
            pass
    return ssl.create_default_context()


def _execute_llm_request(*, req: request.Request, timeout_seconds: float) -> dict:
    max_attempts = max(1, _read_int_env("P1_LLM_MAX_RETRIES", 2) + 1)
    retry_sleep_seconds = max(1.0, _read_float_env("P1_LLM_RETRY_SLEEP_SECONDS", 8.0))
    last_error: RuntimeError | None = None

    for attempt_index in range(max_attempts):
        try:
            with request.urlopen(req, timeout=timeout_seconds, context=_build_ssl_context()) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            runtime_error = RuntimeError(f"http_{exc.code}:{body[:240]}")
            if exc.code == 429 and attempt_index < max_attempts - 1:
                time.sleep(retry_sleep_seconds * (attempt_index + 1))
                last_error = runtime_error
                continue
            raise runtime_error from exc
        except error.URLError as exc:
            runtime_error = RuntimeError(f"url_error:{exc.reason}")
            if attempt_index < max_attempts - 1:
                time.sleep(retry_sleep_seconds * (attempt_index + 1))
                last_error = runtime_error
                continue
            raise runtime_error from exc
        except TimeoutError as exc:
            runtime_error = RuntimeError("timeout_error")
            if attempt_index < max_attempts - 1:
                time.sleep(retry_sleep_seconds * (attempt_index + 1))
                last_error = runtime_error
                continue
            raise runtime_error from exc
        except socket.timeout as exc:
            runtime_error = RuntimeError("socket_timeout")
            if attempt_index < max_attempts - 1:
                time.sleep(retry_sleep_seconds * (attempt_index + 1))
                last_error = runtime_error
                continue
            raise runtime_error from exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("llm_request_failed")


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _build_responses_input(messages: list[dict[str, str]]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for message in messages:
        payload.append(
            {
                "role": message["role"],
                "content": [
                    {
                        "type": "input_text",
                        "text": message["content"],
                    }
                ],
            }
        )
    return payload


def _extract_json_from_chat_completion(response_json: dict) -> dict[str, str | None]:
    choices = response_json.get("choices") or []
    if not choices:
        raise RuntimeError("empty_choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    elif isinstance(content, str):
        text = content
    else:
        raise RuntimeError("missing_message_content")
    raw = _extract_first_json_object(text)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid_json:{exc.msg}") from exc
    return {
        "subject": _coerce_optional_string(parsed.get("subject")),
        "relation": _coerce_optional_string(parsed.get("relation")),
        "object": _coerce_optional_string(parsed.get("object")),
        "qualifier": _coerce_optional_string(parsed.get("qualifier")),
        "time": _coerce_optional_string(parsed.get("time")),
    }


def _extract_batch_json_from_chat_completion(response_json: dict) -> dict[str, dict[str, str | None] | None]:
    choices = response_json.get("choices") or []
    if not choices:
        raise RuntimeError("empty_choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    elif isinstance(content, str):
        text = content
    else:
        raise RuntimeError("missing_message_content")
    return _extract_batch_items_from_text(text)


def _extract_json_from_responses_api(response_json: dict) -> dict[str, str | None]:
    text = _extract_text_from_responses_api(response_json)
    raw = _extract_first_json_object(text)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid_json:{exc.msg}") from exc
    return {
        "subject": _coerce_optional_string(parsed.get("subject")),
        "relation": _coerce_optional_string(parsed.get("relation")),
        "object": _coerce_optional_string(parsed.get("object")),
        "qualifier": _coerce_optional_string(parsed.get("qualifier")),
        "time": _coerce_optional_string(parsed.get("time")),
    }


def _extract_batch_json_from_responses_api(response_json: dict) -> dict[str, dict[str, str | None] | None]:
    text = _extract_text_from_responses_api(response_json)
    return _extract_batch_items_from_text(text)


def _extract_text_from_responses_api(response_json: dict) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    fragments: list[str] = []
    for item in response_json.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content_item in item.get("content") or []:
            if not isinstance(content_item, dict):
                continue
            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                fragments.append(text)
            elif isinstance(text, dict):
                value = text.get("value")
                if isinstance(value, str) and value.strip():
                    fragments.append(value)
    if fragments:
        return "\n".join(fragments)
    raise RuntimeError("missing_responses_output_text")


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("json_object_not_found")
    return text[start : end + 1]


def _extract_batch_items_from_text(text: str) -> dict[str, dict[str, str | None] | None]:
    raw = _extract_first_json_object(text)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid_json:{exc.msg}") from exc
    items = parsed.get("items")
    if not isinstance(items, list):
        raise RuntimeError("missing_batch_items")
    normalized: dict[str, dict[str, str | None] | None] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = _coerce_optional_string(item.get("item_id"))
        if not item_id:
            continue
        normalized[item_id] = {
            "subject": _coerce_optional_string(item.get("subject")),
            "relation": _coerce_optional_string(item.get("relation")),
            "object": _coerce_optional_string(item.get("object")),
            "qualifier": _coerce_optional_string(item.get("qualifier")),
            "time": _coerce_optional_string(item.get("time")),
        }
    return normalized


def _coerce_optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() == "null":
            return None
        return cleaned
    return str(value).strip() or None


def _build_llm_cache_key(
    *,
    model: str,
    base_url: str,
    api_style: str,
    prompt_variant: str,
    text: str,
    dataset: str,
    role: str,
) -> str:
    raw = json.dumps(
        {
            "model": model,
            "base_url": base_url,
            "api_style": api_style,
            "prompt_variant": prompt_variant,
            "text": text,
            "dataset": dataset,
            "role": role,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return sha256(raw.encode("utf-8")).hexdigest()


def _llm_cache_dir() -> Path:
    configured = os.getenv("P1_LLM_CACHE_DIR")
    if configured:
        return Path(configured)
    return Path("data/cache/llm_claims")


def _read_llm_cache(cache_key: str) -> dict[str, str | None] | None:
    cache_path = _llm_cache_dir() / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {
        "subject": _coerce_optional_string(payload.get("subject")),
        "relation": _coerce_optional_string(payload.get("relation")),
        "object": _coerce_optional_string(payload.get("object")),
        "qualifier": _coerce_optional_string(payload.get("qualifier")),
        "time": _coerce_optional_string(payload.get("time")),
    }


def _write_llm_cache(cache_key: str, payload: dict[str, str | None]) -> None:
    cache_dir = _llm_cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{cache_key}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        return
