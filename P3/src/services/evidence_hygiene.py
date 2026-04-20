"""Lightweight evidence-hygiene heuristics for P1-facing exports."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import RetrievalConfig
from src.schemas.retrieval import RetrievalResponse, RetrievedEvidence
from src.utils.text import simple_tokenize

VERB_HINTS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "released",
    "directed",
    "written",
    "premiered",
    "announced",
    "formed",
    "constructed",
    "serves",
    "serving",
    "hosts",
    "stars",
    "based",
    "owns",
    "spent",
    "started",
    "ended",
}
ATTRIBUTION_PREFIXES = (
    "according to",
    "featuring",
    "starring",
    "written and directed by",
    "directed by",
    "born ",
    "for the ",
)
MARKER_TOKENS = ("-LRB-", "-RRB-", "-LSB-", "-RSB-")


@dataclass
class HygieneAssessment:
    """Summary of evidence quality heuristics for one retrieved chunk."""

    penalty: float
    adjusted_score: float
    flags: list[str]
    skip: bool


def select_retrieval_score(evidence: RetrievedEvidence) -> float:
    """Select the stable handoff score with a fixed precedence order."""
    return float(
        evidence.score_rerank
        or evidence.score_hybrid
        or evidence.score_dense
        or evidence.score_sparse
        or 0.0
    )


def assess_evidence_hygiene(evidence: RetrievedEvidence, config: RetrievalConfig) -> HygieneAssessment:
    """Compute a lightweight penalty for extractor-hostile evidence."""
    text = (evidence.text or "").strip()
    tokens = simple_tokenize(text)
    lower_tokens = [token.lower() for token in tokens]
    alpha_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
    alpha_lower_tokens = [token.lower() for token in alpha_tokens]
    token_count = len(tokens)
    verb_count = sum(token in VERB_HINTS for token in alpha_lower_tokens)
    titlecase_tokens = sum(token[:1].isupper() for token in alpha_tokens if token)
    titlecase_ratio = titlecase_tokens / max(len(alpha_tokens), 1)
    comma_count = text.count(",")
    marker_count = sum(text.count(marker) for marker in MARKER_TOKENS)

    penalty = 0.0
    flags: list[str] = []

    ultra_short = token_count < config.evidence_hygiene_min_tokens and verb_count == 0
    if ultra_short:
        penalty += 0.55
        flags.append("ultra_short_non_propositional")

    title_like = token_count <= 8 and titlecase_ratio >= 0.55 and verb_count == 0
    if title_like:
        penalty += 0.35
        flags.append("title_or_header_like")

    enumeration_heavy = comma_count >= 4 and titlecase_ratio >= 0.45 and verb_count <= 2
    if enumeration_heavy:
        penalty += 0.25
        flags.append("enumeration_heavy")

    parenthetical_heavy = marker_count >= 2 or (marker_count / max(token_count, 1)) >= 0.12
    if parenthetical_heavy:
        penalty += 0.2
        flags.append("parenthetical_heavy")

    lower_text = text.lower()
    attribution_heavy = any(lower_text.startswith(prefix) for prefix in ATTRIBUTION_PREFIXES)
    if attribution_heavy:
        penalty += 0.15
        flags.append("attribution_or_appositive_heavy")

    appositive_entity_heavy = verb_count == 0 and comma_count >= 2 and token_count <= 14
    if appositive_entity_heavy:
        penalty += 0.25
        flags.append("entity_or_appositive_heavy")

    penalty = min(round(penalty, 4), 1.0)
    base_score = select_retrieval_score(evidence)
    adjusted_score = round(base_score - (config.evidence_hygiene_penalty_weight * penalty), 6)
    skip = penalty >= config.evidence_hygiene_skip_threshold and any(
        flag in {"ultra_short_non_propositional", "title_or_header_like", "entity_or_appositive_heavy"}
        for flag in flags
    )
    return HygieneAssessment(
        penalty=penalty,
        adjusted_score=adjusted_score,
        flags=flags,
        skip=skip,
    )


def apply_evidence_hygiene(
    response: RetrievalResponse,
    config: RetrievalConfig,
    *,
    top_k: int,
) -> RetrievalResponse:
    """Re-rank retrieved evidence with light extractor-compatibility penalties."""
    if not config.enable_evidence_hygiene:
        return response

    assessed_results: list[tuple[RetrievedEvidence, HygieneAssessment]] = []
    for result in response.results:
        assessment = assess_evidence_hygiene(result, config)
        updated_metadata = {
            **result.metadata,
            "hygiene_penalty": assessment.penalty,
            "hygiene_adjusted_score": assessment.adjusted_score,
            "hygiene_flags": assessment.flags,
            "hygiene_skip_candidate": assessment.skip,
        }
        assessed_results.append((result.model_copy(update={"metadata": updated_metadata}), assessment))

    assessed_results.sort(
        key=lambda item: (
            item[1].skip,
            -item[1].adjusted_score,
            item[0].rank,
        )
    )

    kept = [item[0] for item in assessed_results if not item[1].skip]
    skipped = [item[0] for item in assessed_results if item[1].skip]
    selected = kept[:top_k]
    if len(selected) < top_k:
        selected.extend(skipped[: top_k - len(selected)])

    for rank, result in enumerate(selected, start=1):
        result.rank = rank

    return RetrievalResponse(query=response.query, mode=response.mode, results=selected)
