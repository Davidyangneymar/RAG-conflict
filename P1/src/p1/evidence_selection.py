"""Claim-Conditioned Evidence Selector (CCES).

Existing FNC-1 evidence selectors (`best_sentence`, `top2_span`,
`top3_span`) score every body sentence against the headline, then pick
either the single best sentence or the best contiguous span. Both
strategies have a known failure mode: when the headline references
several distinct facets (e.g. *who* did *what* *when*), the best span
often repeats the same facet because adjacent sentences in news bodies
tend to elaborate the same point.

CCES generalises evidence selection along three axes:

1. **Claim-conditioned scoring** — each candidate sentence is scored
   against the *claim* (headline) directly, with a pluggable similarity
   backend (lexical token-set cosine OR sentence embeddings).
2. **Redundancy-aware selection via Maximal Marginal Relevance (MMR;
   Carbonell & Goldstein, 1998)** — after picking the top sentence we
   penalise candidates that overlap with what we already chose, so the
   final evidence covers diverse facets of the claim.
3. **Backend agnostic** — the similarity function is a pure dependency
   so we can swap in a stronger embedding model without touching the
   selection logic. The default lexical backend has zero dependencies
   and is deterministic.

The output restores original document order so the downstream NLI model
sees a coherent passage. We also expose the per-sentence MMR trace as
metadata for debugging / paper-figure reproduction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Protocol


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
TOKEN_RE = re.compile(r"\b[a-zA-Z0-9]+\b")
STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "in", "on", "for", "of", "to",
        "that", "this", "with", "by", "is", "was", "are", "were", "as",
        "at", "from", "be", "been", "being", "it", "its", "his", "her",
        "their", "they", "them", "we", "our", "you", "your", "i",
    }
)


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------
@dataclass
class SelectedEvidence:
    """The result of evidence selection for a single (claim, document) pair."""

    text: str
    sentences: list[str]
    selected_indices: list[int]
    relevance_scores: list[float]
    mmr_trace: list[dict] = field(default_factory=list)
    backend: str = "lexical"


class SimilarityBackend(Protocol):
    """Compute pairwise similarities between a query and a list of texts."""

    name: str

    def query_to_candidates(self, query: str, candidates: list[str]) -> list[float]:
        ...

    def candidate_to_candidate(self, left: str, right: str) -> float:
        ...


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _coverage(query: set[str], candidate: set[str]) -> float:
    if not query:
        return 0.0
    return len(query & candidate) / len(query)


@dataclass
class LexicalSimilarityBackend:
    """Token-set similarity: 0.6 * coverage(query→cand) + 0.4 * jaccard.

    Coverage rewards sentences that contain *the claim's* keywords
    (asymmetric — what we actually want for evidence retrieval), while
    jaccard tempers reward for very long sentences that contain
    everything by accident.
    """

    name: str = "lexical"
    coverage_weight: float = 0.6
    jaccard_weight: float = 0.4

    def query_to_candidates(self, query: str, candidates: list[str]) -> list[float]:
        query_tokens = _tokenize(query)
        return [self._score(query_tokens, _tokenize(candidate)) for candidate in candidates]

    def candidate_to_candidate(self, left: str, right: str) -> float:
        return _jaccard(_tokenize(left), _tokenize(right))

    def _score(self, query_tokens: set[str], candidate_tokens: set[str]) -> float:
        return (
            self.coverage_weight * _coverage(query_tokens, candidate_tokens)
            + self.jaccard_weight * _jaccard(query_tokens, candidate_tokens)
        )


@dataclass
class EmbeddingSimilarityBackend:
    """Cosine similarity over sentence embeddings.

    Lazily loads `sentence-transformers` so the module import remains
    free of heavyweight dependencies. Falls back to lexical similarity
    if the optional dependency is missing.
    """

    model_name: str = "BAAI/bge-small-en-v1.5"
    name: str = field(init=False)
    _model: object | None = field(default=None, init=False, repr=False)
    _fallback: LexicalSimilarityBackend = field(default_factory=LexicalSimilarityBackend, init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = f"embedding:{self.model_name}"

    def _ensure_model(self) -> object | None:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
            return self._model
        except Exception:  # noqa: BLE001 — any failure routes to fallback
            self._model = None
            return None

    def query_to_candidates(self, query: str, candidates: list[str]) -> list[float]:
        model = self._ensure_model()
        if model is None or not candidates:
            return self._fallback.query_to_candidates(query, candidates)
        embeddings = model.encode([query] + candidates, normalize_embeddings=True)  # type: ignore[attr-defined]
        query_vec = embeddings[0]
        return [float(_dot(query_vec, embeddings[i + 1])) for i in range(len(candidates))]

    def candidate_to_candidate(self, left: str, right: str) -> float:
        model = self._ensure_model()
        if model is None:
            return self._fallback.candidate_to_candidate(left, right)
        embeddings = model.encode([left, right], normalize_embeddings=True)  # type: ignore[attr-defined]
        return float(_dot(embeddings[0], embeddings[1]))


def _dot(left, right) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------
@dataclass
class ClaimConditionedEvidenceSelector:
    """Pick `k` evidence sentences via MMR over claim-conditioned scores.

    Args:
        k: number of sentences to select.
        lambda_param: trade-off in MMR between relevance (sim to claim)
            and diversity (anti-redundancy). λ=1 reduces to top-k by
            relevance; λ=0 reduces to a pure diversity sampler.
        min_relevance: candidates with claim-similarity below this are
            skipped (avoids selecting a low-signal sentence purely
            because it is dissimilar to the others).
        sentence_splitter: regex / function for splitting the document.
        backend: similarity backend (defaults to lexical, no deps).
        keep_original_order: if True the final text concatenates the
            picked sentences in their original document order. If False,
            we use the MMR pick order (highest-relevance-first).
    """

    k: int = 2
    lambda_param: float = 0.7
    min_relevance: float = 0.05
    backend: SimilarityBackend = field(default_factory=LexicalSimilarityBackend)
    keep_original_order: bool = True
    sentence_splitter: Callable[[str], list[str]] = field(default=lambda text: _split_sentences(text))

    def select(self, claim: str, document: str) -> SelectedEvidence:
        sentences = self.sentence_splitter(document)
        if not sentences:
            return SelectedEvidence(
                text=document.strip(),
                sentences=[],
                selected_indices=[],
                relevance_scores=[],
                backend=self.backend.name,
            )
        if len(sentences) <= self.k:
            return SelectedEvidence(
                text=" ".join(sentences),
                sentences=sentences,
                selected_indices=list(range(len(sentences))),
                relevance_scores=self.backend.query_to_candidates(claim, sentences),
                backend=self.backend.name,
            )

        relevance = self.backend.query_to_candidates(claim, sentences)
        selected: list[int] = []
        trace: list[dict] = []
        remaining = set(range(len(sentences)))

        while len(selected) < self.k and remaining:
            best_idx = -1
            best_mmr = float("-inf")
            best_diversity = 0.0
            best_relevance = 0.0
            for idx in remaining:
                rel = relevance[idx]
                if rel < self.min_relevance and selected:
                    continue
                if not selected:
                    diversity_penalty = 0.0
                else:
                    diversity_penalty = max(
                        self.backend.candidate_to_candidate(sentences[idx], sentences[chosen])
                        for chosen in selected
                    )
                mmr = self.lambda_param * rel - (1.0 - self.lambda_param) * diversity_penalty
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
                    best_diversity = diversity_penalty
                    best_relevance = rel
            if best_idx < 0:
                break
            selected.append(best_idx)
            remaining.discard(best_idx)
            trace.append(
                {
                    "step": len(selected),
                    "selected_index": best_idx,
                    "relevance": round(best_relevance, 4),
                    "diversity_penalty": round(best_diversity, 4),
                    "mmr_score": round(best_mmr, 4),
                }
            )

        if not selected:
            # Degenerate: nothing crossed the relevance floor — fall back
            # to the single highest-relevance sentence rather than empty.
            best_idx = max(range(len(sentences)), key=lambda i: relevance[i])
            selected = [best_idx]
            trace.append(
                {
                    "step": 1,
                    "selected_index": best_idx,
                    "relevance": round(relevance[best_idx], 4),
                    "diversity_penalty": 0.0,
                    "mmr_score": round(self.lambda_param * relevance[best_idx], 4),
                    "fallback": "min_relevance_floor_unmet",
                }
            )

        ordered = sorted(selected) if self.keep_original_order else selected
        text = " ".join(sentences[idx] for idx in ordered)
        return SelectedEvidence(
            text=text,
            sentences=sentences,
            selected_indices=ordered,
            relevance_scores=relevance,
            mmr_trace=trace,
            backend=self.backend.name,
        )


def _split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------
def build_evidence_selector(
    *,
    k: int = 2,
    lambda_param: float = 0.7,
    backend: str = "lexical",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    min_relevance: float = 0.05,
    keep_original_order: bool = True,
) -> ClaimConditionedEvidenceSelector:
    """Build a CCES selector with one of the named backends.

    Backends:
        - "lexical": deterministic, zero-dep, fast (default)
        - "embedding": sentence-transformers cosine; falls back to
          lexical if the optional dep is missing
    """
    if backend == "lexical":
        sim_backend: SimilarityBackend = LexicalSimilarityBackend()
    elif backend in {"embedding", "bge", "sentence-transformer"}:
        sim_backend = EmbeddingSimilarityBackend(model_name=embedding_model)
    else:
        raise ValueError(f"Unknown evidence-selector backend: {backend}")
    return ClaimConditionedEvidenceSelector(
        k=k,
        lambda_param=lambda_param,
        min_relevance=min_relevance,
        backend=sim_backend,
        keep_original_order=keep_original_order,
    )
