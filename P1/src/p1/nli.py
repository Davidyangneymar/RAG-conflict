from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from p1.schemas import ClaimPair, NLIPairResult, NliLabel


class NLIModel(Protocol):
    def predict(self, pair: ClaimPair) -> NLIPairResult:
        ...

    def predict_many(self, pairs: list[ClaimPair]) -> list[NLIPairResult]:
        ...


NEGATION_HINTS = {"no", "not", "never", "false", "deny", "denied", "without"}
REFUTATION_HINTS = {"fake", "hoax", "false", "debunked", "denied"}
SUPPORTED_NEGATION_PATTERNS = {
    ("not", "denied"),
    ("no", "false"),
    ("fake", "hoax"),
    ("false", "hoax"),
}


@dataclass
class HeuristicNLIModel:
    bidirectional: bool = True

    def predict(self, pair: ClaimPair) -> NLIPairResult:
        return self.predict_many([pair])[0]

    def predict_many(self, pairs: list[ClaimPair]) -> list[NLIPairResult]:
        """Score claim pairs with deterministic lexical and negation heuristics."""
        results: list[NLIPairResult] = []
        for pair in pairs:
            if self.bidirectional:
                results.append(
                    _merge_bidirectional_predictions(
                        pair,
                        self._predict_once(pair.claim_a.text, pair.claim_b.text, pair.lexical_similarity),
                        self._predict_once(pair.claim_b.text, pair.claim_a.text, pair.lexical_similarity),
                        model_name="heuristic",
                        batch_size=len(pairs),
                    )
                )
                continue

            scores = self._predict_once(pair.claim_a.text, pair.claim_b.text, pair.lexical_similarity)
            label, normalized_scores = _scores_to_label(scores)
            results.append(
                NLIPairResult(
                    claim_a_id=pair.claim_a.claim_id,
                    claim_b_id=pair.claim_b.claim_id,
                    entailment_score=normalized_scores[0],
                    contradiction_score=normalized_scores[1],
                    neutral_score=normalized_scores[2],
                    label=label,
                    is_bidirectional=False,
                    metadata={"model": "heuristic", "nli_batch_size": len(pairs)},
                )
            )
        return results

    def _predict_once(self, left_text: str, right_text: str, overlap: float) -> tuple[float, float, float]:
        left_text = left_text.lower()
        right_text = right_text.lower()
        overlap = overlap

        left_tokens = set(left_text.split())
        right_tokens = set(right_text.split())
        left_neg = any(token in left_tokens for token in NEGATION_HINTS)
        right_neg = any(token in right_tokens for token in NEGATION_HINTS)
        left_refute = any(token in left_tokens for token in REFUTATION_HINTS)
        right_refute = any(token in right_tokens for token in REFUTATION_HINTS)

        if overlap >= 0.18 and _supported_negation_case(left_tokens, right_tokens):
            label = NliLabel.ENTAILMENT
            scores = (0.76, 0.12, 0.12)
        elif overlap >= 0.22 and ((left_refute and not right_refute) or (right_refute and not left_refute)):
            label = NliLabel.CONTRADICTION
            scores = (0.08, 0.78, 0.14)
        elif overlap >= 0.3 and left_neg != right_neg:
            label = NliLabel.CONTRADICTION
            scores = (0.08, 0.82, 0.10)
        elif overlap >= 0.4:
            scores = (0.82, 0.08, 0.10)
        else:
            scores = (0.10, 0.10, 0.80)
        return scores


def _supported_negation_case(left_tokens: set[str], right_tokens: set[str]) -> bool:
    for left_hint, right_hint in SUPPORTED_NEGATION_PATTERNS:
        if (left_hint in left_tokens and right_hint in right_tokens) or (right_hint in left_tokens and left_hint in right_tokens):
            return True
    return False


class HuggingFaceCrossEncoderNLI:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large", bidirectional: bool = True):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("transformers is required for HuggingFaceCrossEncoderNLI") from exc

        resolved_model = _resolve_model_path(model_name)
        self.model_name = resolved_model
        self.bidirectional = bidirectional
        self._pipeline = pipeline("text-classification", model=resolved_model, top_k=None)

    def predict(self, pair: ClaimPair) -> NLIPairResult:
        return self.predict_many([pair])[0]

    def predict_many(self, pairs: list[ClaimPair]) -> list[NLIPairResult]:
        """Run the HuggingFace cross-encoder over a batch of claim pairs."""
        if not pairs:
            return []

        forward_scores_list = self._predict_many_once(
            [(pair.claim_a.text, pair.claim_b.text) for pair in pairs]
        )
        if self.bidirectional:
            backward_scores_list = self._predict_many_once(
                [(pair.claim_b.text, pair.claim_a.text) for pair in pairs]
            )
            return [
                _merge_bidirectional_predictions(
                    pair,
                    forward_scores,
                    backward_scores,
                    model_name=self.model_name,
                    batch_size=len(pairs),
                )
                for pair, forward_scores, backward_scores in zip(pairs, forward_scores_list, backward_scores_list)
            ]

        results: list[NLIPairResult] = []
        for pair, forward_scores in zip(pairs, forward_scores_list):
            top_label, normalized_scores = _scores_to_label(forward_scores)
            results.append(
                NLIPairResult(
                    claim_a_id=pair.claim_a.claim_id,
                    claim_b_id=pair.claim_b.claim_id,
                    entailment_score=normalized_scores[0],
                    contradiction_score=normalized_scores[1],
                    neutral_score=normalized_scores[2],
                    label=top_label,
                    is_bidirectional=False,
                    metadata={"model": self.model_name, "nli_batch_size": len(pairs)},
                )
            )
        return results

    def _predict_once(self, left_text: str, right_text: str) -> tuple[float, float, float]:
        raw_outputs = self._pipeline({"text": left_text, "text_pair": right_text})
        scores = {item["label"].lower(): float(item["score"]) for item in raw_outputs}
        return (
            scores.get("entailment", 0.0),
            scores.get("contradiction", 0.0),
            scores.get("neutral", 0.0),
        )

    def _predict_many_once(self, text_pairs: list[tuple[str, str]]) -> list[tuple[float, float, float]]:
        raw_outputs_list = self._pipeline(
            [{"text": left_text, "text_pair": right_text} for left_text, right_text in text_pairs]
        )
        if isinstance(raw_outputs_list, dict):
            raw_outputs_list = [raw_outputs_list]
        scores_list: list[tuple[float, float, float]] = []
        for raw_outputs in raw_outputs_list:
            if isinstance(raw_outputs, dict):
                raw_outputs = [raw_outputs]
            scores = {item["label"].lower(): float(item["score"]) for item in raw_outputs}
            scores_list.append(
                (
                    scores.get("entailment", 0.0),
                    scores.get("contradiction", 0.0),
                    scores.get("neutral", 0.0),
                )
            )
        return scores_list


def build_nli_model(
    kind: str = "heuristic",
    model_name: str = "cross-encoder/nli-deberta-v3-large",
    bidirectional: bool = True,
) -> NLIModel:
    """Construct an NLI backend by kind.

    Supported kinds:
        - "heuristic": deterministic lexical+negation baseline
        - "hf" / "huggingface" / "cross-encoder": HuggingFace cross-encoder
        - "llm": zero-shot LLM via the responses/chat API (P1_LLM_* env vars)
    """
    normalized = kind.strip().lower()
    if normalized == "heuristic":
        return HeuristicNLIModel(bidirectional=bidirectional)
    if normalized in {"hf", "huggingface", "cross-encoder"}:
        return HuggingFaceCrossEncoderNLI(model_name=model_name, bidirectional=bidirectional)
    if normalized in {"llm", "llm-zero-shot", "zero-shot"}:
        from p1.llm_nli import LLMNLIModel
        return LLMNLIModel(bidirectional=bidirectional)
    raise ValueError(f"Unsupported NLI model kind: {kind}")


def _resolve_model_path(model_name: str) -> str:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return model_name


def _scores_to_label(scores: tuple[float, float, float]) -> tuple[NliLabel, tuple[float, float, float]]:
    entailment, contradiction, neutral = scores
    top_label = max(
        {
            NliLabel.ENTAILMENT: entailment,
            NliLabel.CONTRADICTION: contradiction,
            NliLabel.NEUTRAL: neutral,
        }.items(),
        key=lambda item: item[1],
    )[0]
    return top_label, scores


def _merge_bidirectional_predictions(
    pair: ClaimPair,
    forward_scores: tuple[float, float, float],
    backward_scores: tuple[float, float, float],
    model_name: str,
    batch_size: int,
) -> NLIPairResult:
    merged_scores = tuple(round((left + right) / 2, 6) for left, right in zip(forward_scores, backward_scores))
    label, normalized_scores = _scores_to_label(merged_scores)
    return NLIPairResult(
        claim_a_id=pair.claim_a.claim_id,
        claim_b_id=pair.claim_b.claim_id,
        entailment_score=normalized_scores[0],
        contradiction_score=normalized_scores[1],
        neutral_score=normalized_scores[2],
        label=label,
        is_bidirectional=True,
        metadata={
            "model": model_name,
            "forward_scores": forward_scores,
            "backward_scores": backward_scores,
            "nli_batch_size": batch_size,
        },
    )
