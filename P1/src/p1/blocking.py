from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from p1.schemas import Claim, ClaimPair


TOKEN_RE = re.compile(r"\b[a-zA-Z0-9]+\b")
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "in",
    "on",
    "for",
    "of",
    "to",
    "that",
    "this",
    "with",
    "by",
    "is",
    "was",
    "are",
    "were",
}


class Embedder(Protocol):
    def encode(self, text: str) -> list[float]:
        ...

    def similarity(self, left: str, right: str) -> float:
        ...


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass
class BlockingConfig:
    min_entity_overlap: int = 1
    min_lexical_similarity: float | None = 0.1
    query_pair_min_lexical_similarity: float | None = None
    min_embedding_similarity: float | None = None
    allow_empty_entities: bool = True
    combine_mode: str = "cascade"


@dataclass
class MultiStageBlocker:
    config: BlockingConfig = field(default_factory=BlockingConfig)
    embedder: Embedder | None = None

    def generate_pairs(self, claims: list[Claim]) -> list[ClaimPair]:
        """Generate candidate claim pairs that pass entity, lexical, and embedding gates."""
        pairs: list[ClaimPair] = []
        if self.embedder is not None:
            for claim in claims:
                if not claim.embedding:
                    claim.embedding = self.embedder.encode(claim.text)

        for claim_a, claim_b in itertools.combinations(claims, 2):
            pair = self._build_pair(claim_a, claim_b)
            if pair is not None:
                pairs.append(pair)

        return pairs

    def _build_pair(self, claim_a: Claim, claim_b: Claim) -> ClaimPair | None:
        entity_overlap = sorted(set(claim_a.entities) & set(claim_b.entities))
        entity_pass = self._passes_entity_stage(claim_a, claim_b, entity_overlap)

        lexical_similarity = _jaccard(_tokenize(claim_a.text), _tokenize(claim_b.text))
        lexical_pass = self._passes_lexical_stage(claim_a, claim_b, lexical_similarity)

        embedding_similarity = None
        embedding_pass = True
        if self.embedder is not None:
            if not claim_a.embedding:
                claim_a.embedding = self.embedder.encode(claim_a.text)
            if not claim_b.embedding:
                claim_b.embedding = self.embedder.encode(claim_b.text)
            embedding_similarity = _cosine(claim_a.embedding, claim_b.embedding)
            embedding_pass = self._passes_embedding_stage(embedding_similarity)

        if not self._passes_combined_stage(entity_pass, lexical_pass, embedding_pass):
            return None

        return ClaimPair(
            claim_a=claim_a,
            claim_b=claim_b,
            entity_overlap=entity_overlap,
            lexical_similarity=round(lexical_similarity, 4),
            embedding_similarity=round(embedding_similarity, 4) if embedding_similarity is not None else None,
        )

    def _passes_entity_stage(self, claim_a: Claim, claim_b: Claim, entity_overlap: list[str]) -> bool:
        if not entity_overlap and not self.config.allow_empty_entities:
            return False
        if claim_a.entities and claim_b.entities:
            return len(entity_overlap) >= self.config.min_entity_overlap
        return self.config.allow_empty_entities

    def _passes_lexical_stage(self, claim_a: Claim, claim_b: Claim, lexical_similarity: float) -> bool:
        threshold = self.config.min_lexical_similarity
        if _is_query_evidence_pair(claim_a, claim_b) and self.config.query_pair_min_lexical_similarity is not None:
            threshold = self.config.query_pair_min_lexical_similarity
        if threshold is None:
            return True
        return lexical_similarity >= threshold

    def _passes_embedding_stage(self, embedding_similarity: float) -> bool:
        threshold = self.config.min_embedding_similarity
        if threshold is None:
            return True
        return embedding_similarity >= threshold

    def _passes_combined_stage(self, entity_pass: bool, lexical_pass: bool, embedding_pass: bool) -> bool:
        mode = self.config.combine_mode.strip().lower()
        if mode == "cascade":
            return entity_pass and lexical_pass and embedding_pass
        if mode == "union":
            return lexical_pass and (entity_pass or embedding_pass)
        raise ValueError(f"Unsupported blocking combine mode: {self.config.combine_mode}")


class CosineVectorEmbedder:
    def __init__(self, encoder: object):
        self.encoder = encoder

    def encode(self, text: str) -> list[float]:
        vector = self.encoder.encode(text)
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)

    def similarity(self, left: str, right: str) -> float:
        left_vector = self.encode(left)
        right_vector = self.encode(right)
        return _cosine(left_vector, right_vector)


class SentenceTransformersEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for SentenceTransformersEmbedder") from exc

        resolved_model = _resolve_model_path(model_name)
        self.model_name = resolved_model
        self.encoder = SentenceTransformer(resolved_model)

    def encode(self, text: str) -> list[float]:
        vector = self.encoder.encode(text, normalize_embeddings=True)
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)

    def similarity(self, left: str, right: str) -> float:
        left_vector = self.encode(left)
        right_vector = self.encode(right)
        return _cosine(left_vector, right_vector)


class TransformersMeanPoolEmbedder:
    def __init__(self, model_name: str):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for TransformersMeanPoolEmbedder") from exc

        resolved_model = _resolve_model_path(model_name)
        self.model_name = resolved_model
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_model)
        self.model = AutoModel.from_pretrained(resolved_model)
        self.model.eval()

    def encode(self, text: str) -> list[float]:
        with self.torch.inference_mode():
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = summed / counts
            normalized = self.torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        vector = normalized[0].cpu().tolist()
        return vector

    def similarity(self, left: str, right: str) -> float:
        left_vector = self.encode(left)
        right_vector = self.encode(right)
        return _cosine(left_vector, right_vector)


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _resolve_model_path(model_name: str) -> str:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return model_name


def _is_query_evidence_pair(claim_a: Claim, claim_b: Claim) -> bool:
    role_a = str(claim_a.source.metadata.get("role") or "").strip().lower()
    role_b = str(claim_b.source.metadata.get("role") or "").strip().lower()
    return {role_a, role_b} == {"query", "retrieved_evidence"}
