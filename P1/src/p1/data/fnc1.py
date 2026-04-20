from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from p1.claim_extraction import extract_entity_candidates
from p1.evidence_selection import build_evidence_selector
from p1.schemas import Claim, ClaimPair, ClaimSource, NliLabel, RetrievalInput, RetrievedChunk


# Cache selectors so we do not rebuild a (potentially heavyweight) selector
# per sample when iterating large batches.
_CCES_CACHE: dict[tuple, object] = {}


def _get_cces_selector(backend: str, k: int, lambda_param: float):
    key = (backend, k, lambda_param)
    if key not in _CCES_CACHE:
        _CCES_CACHE[key] = build_evidence_selector(
            k=k, lambda_param=lambda_param, backend=backend
        )
    return _CCES_CACHE[key]


def select_cces_evidence(
    headline: str,
    body: str,
    *,
    k: int = 2,
    lambda_param: float = 0.7,
    backend: str = "lexical",
) -> str:
    """Body-mode entry point for the Claim-Conditioned Evidence Selector."""
    selector = _get_cces_selector(backend=backend, k=k, lambda_param=lambda_param)
    return selector.select(headline, body).text  # type: ignore[attr-defined]


STANCE_TO_NLI = {
    "agree": NliLabel.ENTAILMENT.value,
    "disagree": NliLabel.CONTRADICTION.value,
    "discuss": NliLabel.NEUTRAL.value,
    "unrelated": NliLabel.NEUTRAL.value,
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
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
    "as",
    "at",
    "from",
}
NEGATION_HINTS = {"no", "not", "never", "false", "deny", "denied", "without", "fake", "hoax"}
NUMBER_RE = re.compile(r"\b\d+\b")
QUOTE_RE = re.compile(r"[\"'“”‘’]")


def read_bodies(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["Body ID"]: row["articleBody"] for row in reader}


def iter_stances(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def convert_fnc1(bodies_path: str | Path, stances_path: str | Path) -> list[dict[str, str]]:
    bodies = read_bodies(bodies_path)
    rows = iter_stances(stances_path)
    normalized: list[dict[str, str]] = []

    for index, row in enumerate(rows):
        body_id = row["Body ID"]
        stance = row["Stance"].strip().lower()
        normalized.append(
            {
                "sample_id": f"fnc1-{index}",
                "headline": row["Headline"].strip(),
                "body_id": body_id,
                "body": bodies.get(body_id, "").strip(),
                "stance_label": stance,
                "nli_label": STANCE_TO_NLI.get(stance, NliLabel.NEUTRAL.value),
            }
        )

    return normalized


def write_jsonl(records: list[dict[str, str]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_to_claim_pair(
    sample: dict[str, str],
    body_mode: str = "full",
    entity_backend: str = "auto",
) -> ClaimPair:
    sample_id = sample["sample_id"]
    headline = sample["headline"].strip()
    body = sample["body"].strip()
    if body_mode == "best_sentence":
        body = select_best_body_sentence(headline, body)
    elif body_mode == "top2_span":
        body = select_best_evidence_span(headline, body, span_size=2)
    elif body_mode == "top3_span":
        body = select_best_evidence_span(headline, body, span_size=3)
    elif body_mode == "cces":
        body = select_cces_evidence(headline, body, k=2, lambda_param=0.7, backend="lexical")
    elif body_mode == "cces3":
        body = select_cces_evidence(headline, body, k=3, lambda_param=0.7, backend="lexical")
    elif body_mode == "cces_embed":
        body = select_cces_evidence(headline, body, k=2, lambda_param=0.7, backend="embedding")

    claim_a = Claim(
        claim_id=f"{sample_id}:headline",
        text=headline,
        source=ClaimSource(
            doc_id=f"{sample_id}:headline",
            chunk_id="headline",
            metadata={"role": "headline", "stance_label": sample["stance_label"]},
        ),
        source_doc_id=f"{sample_id}:headline",
        source_chunk_id="headline",
        entities=extract_entity_candidates(headline, backend=entity_backend),
        metadata={"dataset": "fnc1"},
    )
    claim_b = Claim(
        claim_id=f"{sample_id}:body",
        text=body,
        source=ClaimSource(
            doc_id=f"body:{sample['body_id']}",
            chunk_id=sample["body_id"],
            metadata={"role": "body", "stance_label": sample["stance_label"]},
        ),
        source_doc_id=f"body:{sample['body_id']}",
        source_chunk_id=sample["body_id"],
        entities=extract_entity_candidates(body, backend=entity_backend),
        metadata={"dataset": "fnc1"},
    )
    return ClaimPair(claim_a=claim_a, claim_b=claim_b)


def sample_to_retrieval_input(
    sample: dict[str, str],
    *,
    body_mode: str = "top2_span",
) -> RetrievalInput:
    body = sample["body"].strip()
    if body_mode == "best_sentence":
        body = select_best_body_sentence(sample["headline"], body)
    elif body_mode == "top2_span":
        body = select_best_evidence_span(sample["headline"], body, span_size=2)
    elif body_mode == "top3_span":
        body = select_best_evidence_span(sample["headline"], body, span_size=3)
    elif body_mode == "cces":
        body = select_cces_evidence(sample["headline"], body, k=2, lambda_param=0.7, backend="lexical")
    elif body_mode == "cces3":
        body = select_cces_evidence(sample["headline"], body, k=3, lambda_param=0.7, backend="lexical")
    elif body_mode == "cces_embed":
        body = select_cces_evidence(sample["headline"], body, k=2, lambda_param=0.7, backend="embedding")

    return RetrievalInput(
        sample_id=sample["sample_id"],
        query=sample["headline"].strip(),
        label=sample["nli_label"],
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id=f"body:{sample['body_id']}",
                text=body,
                rank=1,
                retrieval_score=1.0,
                metadata={
                    "dataset": "fnc1",
                    "stance_label": sample["stance_label"],
                    "body_id": sample["body_id"],
                },
            )
        ],
        metadata={
            "dataset": "fnc1",
            "stance_label": sample["stance_label"],
            "body_id": sample["body_id"],
        },
    )


def select_best_body_sentence(headline: str, body: str) -> str:
    scored = rank_body_sentences(headline, body, top_k=1)
    if not scored:
        return body.strip()
    if float(scored[0]["score"]) < 0.12:
        sentences = _split_sentences(body)
        return sentences[0] if sentences else body.strip()
    return scored[0]["sentence"]


def select_best_evidence_span(headline: str, body: str, span_size: int = 2) -> str:
    sentences = _split_sentences(body)
    if not sentences:
        return body.strip()
    if len(sentences) <= span_size:
        return " ".join(sentences)

    headline_tokens = _tokenize(headline)
    scored = rank_body_sentences(headline, body, top_k=len(sentences))
    score_by_index = {int(item["index"]): float(item["score"]) for item in scored}

    best_start = 0
    best_score = -1.0
    best_overlap = 0.0
    for start in range(0, len(sentences) - span_size + 1):
        indices = range(start, start + span_size)
        joined_text = " ".join(sentences[idx] for idx in indices)
        joined_tokens = _tokenize(joined_text)
        span_score = sum(score_by_index.get(idx, 0.0) for idx in indices)
        span_coverage = _coverage(headline_tokens, joined_tokens)
        span_jaccard = _jaccard(headline_tokens, joined_tokens)
        combined = span_score + (0.25 * span_coverage) + (0.15 * span_jaccard)
        if combined > best_score:
            best_start = start
            best_score = combined
            best_overlap = len(headline_tokens & joined_tokens)

    lead_span = " ".join(sentences[:span_size])
    if best_score < 0.2 or best_overlap == 0:
        return lead_span
    return " ".join(sentences[best_start : best_start + span_size])


def rank_body_sentences(headline: str, body: str, top_k: int = 3) -> list[dict[str, float | str]]:
    sentences = _split_sentences(body)
    if not sentences:
        return [{"sentence": body.strip(), "score": 0.0, "coverage": 0.0, "jaccard": 0.0}]

    headline_variants = _headline_variants(headline)
    headline_numbers = set(NUMBER_RE.findall(headline))
    headline_negation = any(token in _tokenize(headline) for token in NEGATION_HINTS)
    scored_sentences: list[dict[str, float | str]] = []

    for index, sentence in enumerate(sentences):
        sentence_tokens = _tokenize(sentence)
        coverage = 0.0
        jaccard = 0.0
        overlap = 0
        for variant in headline_variants:
            variant_tokens = _tokenize(variant)
            variant_coverage = _coverage(variant_tokens, sentence_tokens)
            variant_jaccard = _jaccard(variant_tokens, sentence_tokens)
            variant_overlap = len(variant_tokens & sentence_tokens)
            if (variant_coverage, variant_jaccard, variant_overlap) > (coverage, jaccard, overlap):
                coverage = variant_coverage
                jaccard = variant_jaccard
                overlap = variant_overlap
        numbers = set(NUMBER_RE.findall(sentence))
        number_bonus = 0.1 if headline_numbers and headline_numbers & numbers else 0.0
        negation_bonus = 0.08 if headline_negation and any(token in sentence_tokens for token in NEGATION_HINTS) else 0.0
        quote_bonus = 0.05 if QUOTE_RE.search(headline) and QUOTE_RE.search(sentence) else 0.0
        colon_bonus = 0.06 if ":" in headline and overlap >= 2 else 0.0
        lead_bias = max(0.0, 0.06 - (0.02 * index))
        low_overlap_penalty = -0.08 if overlap == 0 else (-0.04 if overlap == 1 else 0.0)
        score = (
            (0.65 * coverage)
            + (0.25 * jaccard)
            + number_bonus
            + negation_bonus
            + quote_bonus
            + colon_bonus
            + lead_bias
            + low_overlap_penalty
        )
        scored_sentences.append(
            {
                "index": index,
                "sentence": sentence,
                "score": round(score, 4),
                "coverage": round(coverage, 4),
                "jaccard": round(jaccard, 4),
            }
        )

    scored_sentences.sort(key=lambda item: float(item["score"]), reverse=True)
    return scored_sentences[:top_k]


def _split_sentences(body: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(body) if sentence.strip()]


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _coverage(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _headline_variants(headline: str) -> list[str]:
    variants = [headline.strip()]
    dequoted = QUOTE_RE.sub("", headline).strip()
    if dequoted and dequoted not in variants:
        variants.append(dequoted)

    if ":" in headline:
        left, right = headline.split(":", 1)
        left = left.strip()
        right = right.strip()
        if left and left not in variants:
            variants.append(left)
        if right and right not in variants:
            variants.append(right)
        right_dequoted = QUOTE_RE.sub("", right).strip()
        if right_dequoted and right_dequoted not in variants:
            variants.append(right_dequoted)

    tokens = headline.split()
    if len(tokens) > 4:
        tail = " ".join(tokens[-6:])
        if tail not in variants:
            variants.append(tail)

    return variants
