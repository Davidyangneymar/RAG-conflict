"""Ensemble + calibration wrappers around any `NLIModel`.

These wrappers compose with the existing heuristic / HuggingFace / LLM
backends through the `NLIModel` Protocol — none of the underlying models
need to know they are being combined or post-processed.

Three primitives:
    * `EnsembleNLIModel`       — weighted score fusion across N backends
    * `CalibratedNLIModel`     — temperature scaling on the score triple
    * `ThresholdedNLIModel`    — class-specific minimum margins, falling
                                 back to neutral when no class is decisive

The ensemble is *late fusion*: each backend produces an
`(entailment, contradiction, neutral)` triple, we re-normalize, then take
a weighted average. This is the standard, robust combiner — heterogeneous
errors decorrelate so the ensemble strictly dominates the average member
under mild assumptions (Dietterich, 2000).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Protocol

from p1.schemas import ClaimPair, NLIPairResult, NliLabel


class NLIModel(Protocol):
    def predict(self, pair: ClaimPair) -> NLIPairResult: ...
    def predict_many(self, pairs: list[ClaimPair]) -> list[NLIPairResult]: ...


_LABEL_ORDER: tuple[NliLabel, NliLabel, NliLabel] = (
    NliLabel.ENTAILMENT,
    NliLabel.CONTRADICTION,
    NliLabel.NEUTRAL,
)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
@dataclass
class EnsembleNLIModel:
    """Weighted late-fusion ensemble over heterogeneous NLI backends.

    Args:
        models: list of (model, weight) pairs. Weights are normalized so
            users do not have to.
        name: optional ensemble label written to result metadata.
    """

    models: list[tuple[NLIModel, float]]
    name: str = "ensemble"

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("EnsembleNLIModel requires at least one backend")
        total = sum(weight for _, weight in self.models)
        if total <= 0:
            raise ValueError("ensemble weights must sum to a positive number")
        self._normalized_weights = [weight / total for _, weight in self.models]

    def predict(self, pair: ClaimPair) -> NLIPairResult:
        return self.predict_many([pair])[0]

    def predict_many(self, pairs: list[ClaimPair]) -> list[NLIPairResult]:
        if not pairs:
            return []
        per_model_results: list[list[NLIPairResult]] = [
            model.predict_many(pairs) for model, _ in self.models
        ]

        merged: list[NLIPairResult] = []
        for index, pair in enumerate(pairs):
            ent = con = neu = 0.0
            member_meta: list[dict] = []
            for results, weight in zip(per_model_results, self._normalized_weights):
                result = results[index]
                triple = _normalize_triple(
                    (result.entailment_score, result.contradiction_score, result.neutral_score)
                )
                ent += weight * triple[0]
                con += weight * triple[1]
                neu += weight * triple[2]
                member_meta.append(
                    {
                        "model": str(result.metadata.get("model")) if result.metadata else None,
                        "label": result.label.value,
                        "weight": weight,
                        "scores": triple,
                    }
                )

            triple = _normalize_triple((ent, con, neu))
            label = _scores_to_label(triple)
            merged.append(
                NLIPairResult(
                    claim_a_id=pair.claim_a.claim_id,
                    claim_b_id=pair.claim_b.claim_id,
                    entailment_score=triple[0],
                    contradiction_score=triple[1],
                    neutral_score=triple[2],
                    label=label,
                    is_bidirectional=any(r.is_bidirectional for r in (results[index] for results in per_model_results)),
                    metadata={
                        "model": self.name,
                        "ensemble_members": member_meta,
                        "nli_batch_size": len(pairs),
                    },
                )
            )
        return merged


# ---------------------------------------------------------------------------
# Calibration (temperature scaling)
# ---------------------------------------------------------------------------
@dataclass
class CalibratedNLIModel:
    """Temperature scaling around an inner NLI model.

    The inner model is assumed to return probability-like scores. We
    treat them as logits via log(p), divide by T, and re-softmax. T < 1
    sharpens; T > 1 smooths. Fit T on a dev set with `fit_temperature`
    by minimizing NLL.
    """

    inner: NLIModel
    temperature: float = 1.0

    def predict(self, pair: ClaimPair) -> NLIPairResult:
        return self.predict_many([pair])[0]

    def predict_many(self, pairs: list[ClaimPair]) -> list[NLIPairResult]:
        results = self.inner.predict_many(pairs)
        adjusted: list[NLIPairResult] = []
        for result in results:
            triple = (result.entailment_score, result.contradiction_score, result.neutral_score)
            scaled = _temperature_scale(triple, self.temperature)
            label = _scores_to_label(scaled)
            metadata = dict(result.metadata or {})
            metadata.update({"calibration_temperature": self.temperature, "uncalibrated_scores": triple})
            adjusted.append(
                NLIPairResult(
                    claim_a_id=result.claim_a_id,
                    claim_b_id=result.claim_b_id,
                    entailment_score=scaled[0],
                    contradiction_score=scaled[1],
                    neutral_score=scaled[2],
                    label=label,
                    is_bidirectional=result.is_bidirectional,
                    metadata=metadata,
                )
            )
        return adjusted


def fit_temperature(
    score_triples: list[tuple[float, float, float]],
    gold_labels: list[NliLabel],
    *,
    candidates: tuple[float, ...] = (0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0),
) -> float:
    """Pick the temperature minimizing average negative log likelihood.

    A grid search is enough at this scale (~thousands of samples) and avoids
    a SciPy dependency. Returns the best temperature; defaults to 1.0 when
    inputs are empty or mismatched.
    """
    if not score_triples or len(score_triples) != len(gold_labels):
        return 1.0
    best_temp = 1.0
    best_nll = float("inf")
    for temperature in candidates:
        nll = 0.0
        for triple, gold in zip(score_triples, gold_labels):
            scaled = _temperature_scale(triple, temperature)
            index = _LABEL_ORDER.index(gold)
            prob = max(scaled[index], 1e-12)
            nll += -_safe_log(prob)
        if nll < best_nll:
            best_nll = nll
            best_temp = temperature
    return best_temp


# ---------------------------------------------------------------------------
# Thresholded post-processor
# ---------------------------------------------------------------------------
@dataclass
class ThresholdedNLIModel:
    """Apply class-specific decision thresholds; fall back to neutral.

    Use case: in FNC-1 the unrelated/neutral class dominates, so requiring
    a minimum confidence before predicting entailment / contradiction
    shifts ambiguous cases to neutral and improves macro-F1.

    Thresholds are absolute probabilities on the score triple. Default
    of 0.0 disables thresholding (acts as identity).
    """

    inner: NLIModel
    entailment_threshold: float = 0.0
    contradiction_threshold: float = 0.0
    fallback_label: NliLabel = NliLabel.NEUTRAL

    def predict(self, pair: ClaimPair) -> NLIPairResult:
        return self.predict_many([pair])[0]

    def predict_many(self, pairs: list[ClaimPair]) -> list[NLIPairResult]:
        results = self.inner.predict_many(pairs)
        adjusted: list[NLIPairResult] = []
        for result in results:
            label = result.label
            ent = result.entailment_score
            con = result.contradiction_score
            if label == NliLabel.ENTAILMENT and ent < self.entailment_threshold:
                label = self.fallback_label
            elif label == NliLabel.CONTRADICTION and con < self.contradiction_threshold:
                label = self.fallback_label
            metadata = dict(result.metadata or {})
            metadata.update(
                {
                    "thresholds": {
                        "entailment": self.entailment_threshold,
                        "contradiction": self.contradiction_threshold,
                    },
                    "pre_threshold_label": result.label.value,
                }
            )
            adjusted.append(
                NLIPairResult(
                    claim_a_id=result.claim_a_id,
                    claim_b_id=result.claim_b_id,
                    entailment_score=result.entailment_score,
                    contradiction_score=result.contradiction_score,
                    neutral_score=result.neutral_score,
                    label=label,
                    is_bidirectional=result.is_bidirectional,
                    metadata=metadata,
                )
            )
        return adjusted


def grid_search_thresholds(
    score_triples: list[tuple[float, float, float]],
    gold_labels: list[NliLabel],
    *,
    candidates: tuple[float, ...] = (0.0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7),
) -> tuple[float, float, float]:
    """Sweep entailment/contradiction thresholds, return the pair maximising macro-F1.

    Returns ``(best_ent_threshold, best_con_threshold, best_macro_f1)``.
    """
    if not score_triples or len(score_triples) != len(gold_labels):
        return (0.0, 0.0, 0.0)

    label_values = [label.value for label in _LABEL_ORDER]
    best = (0.0, 0.0, -1.0)
    for ent_threshold in candidates:
        for con_threshold in candidates:
            confusion: dict[str, dict[str, int]] = {
                label: {other: 0 for other in label_values} for label in label_values
            }
            for triple, gold in zip(score_triples, gold_labels):
                pred_index = max(range(3), key=lambda i: triple[i])
                pred_label = _LABEL_ORDER[pred_index]
                if pred_label == NliLabel.ENTAILMENT and triple[0] < ent_threshold:
                    pred_label = NliLabel.NEUTRAL
                elif pred_label == NliLabel.CONTRADICTION and triple[1] < con_threshold:
                    pred_label = NliLabel.NEUTRAL
                confusion[gold.value][pred_label.value] += 1
            macro_f1 = _macro_f1(confusion, label_values)
            if macro_f1 > best[2]:
                best = (ent_threshold, con_threshold, macro_f1)
    return best


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _normalize_triple(triple: tuple[float, float, float]) -> tuple[float, float, float]:
    a, b, c = (max(0.0, x) for x in triple)
    total = a + b + c
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (a / total, b / total, c / total)


def _temperature_scale(triple: tuple[float, float, float], temperature: float) -> tuple[float, float, float]:
    if temperature <= 0:
        temperature = 1.0
    safe = [max(value, 1e-9) for value in triple]
    logits = [_safe_log(value) / temperature for value in safe]
    max_logit = max(logits)
    exps = [exp(value - max_logit) for value in logits]
    total = sum(exps)
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return tuple(round(value / total, 6) for value in exps)  # type: ignore[return-value]


def _safe_log(value: float) -> float:
    from math import log
    return log(max(value, 1e-12))


def _scores_to_label(triple: tuple[float, float, float]) -> NliLabel:
    return _LABEL_ORDER[max(range(3), key=lambda i: triple[i])]


def _macro_f1(confusion: dict[str, dict[str, int]], labels: list[str]) -> float:
    f1_scores: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
