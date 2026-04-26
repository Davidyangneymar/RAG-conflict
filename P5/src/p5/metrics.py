from __future__ import annotations

from dataclasses import dataclass

from .schemas import PredictionRecord

CONTRADICTION_ALIASES = {
    "contradiction",
    "refuted",
    "refutes",
    "disagree",
    "hard_contradiction",
    "misinformation",
}
ENTAILMENT_ALIASES = {
    "entailment",
    "supported",
    "supports",
    "agree",
}
NEUTRAL_ALIASES = {
    "neutral",
    "not enough evidence",
    "not enough info",
    "nei",
    "discuss",
    "unrelated",
    "conflicting",
    "conflicting evidence/cherrypicking",
    "conflicting evidence/cherry-picking",
    "ambiguity",
}
ABSTAIN_ALIASES = {"", "none", "null", "abstain", "unknown", "no_answer", "n/a"}


@dataclass
class Metrics:
    contradiction_f1: float
    stance_macro_f1: float
    abstention_rate: float
    missing_prediction_rate: float
    support: int


def normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    normalized = str(label).strip().lower()
    if normalized in ABSTAIN_ALIASES:
        return None
    if normalized in CONTRADICTION_ALIASES:
        return "contradiction"
    if normalized in ENTAILMENT_ALIASES:
        return "entailment"
    if normalized in NEUTRAL_ALIASES:
        return "neutral"
    return normalized


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _binary_contradiction_f1(gold: list[str], pred: list[str | None]) -> float:
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        g_pos = g == "contradiction"
        p_pos = p == "contradiction"
        if g_pos and p_pos:
            tp += 1
        elif (not g_pos) and p_pos:
            fp += 1
        elif g_pos and (not p_pos):
            fn += 1
    return _f1(tp, fp, fn)


def _macro_f1(gold: list[str], pred: list[str | None], labels: tuple[str, ...]) -> float:
    per_label: list[float] = []
    for label in labels:
        tp = fp = fn = 0
        for g, p in zip(gold, pred):
            if g == label and p == label:
                tp += 1
            elif g != label and p == label:
                fp += 1
            elif g == label and p != label:
                fn += 1
        per_label.append(_f1(tp, fp, fn))
    return sum(per_label) / len(per_label) if per_label else 0.0


def compute_metrics(
    gold_labels: list[str],
    predictions: list[PredictionRecord],
    *,
    total_support: int | None = None,
    missing_prediction_count: int = 0,
) -> Metrics:
    normalized_gold = [normalize_label(v) or "neutral" for v in gold_labels]
    normalized_pred = [normalize_label(p.predicted_label) for p in predictions]

    abstained = sum(1 for p in normalized_pred if p is None)
    contradiction_f1 = _binary_contradiction_f1(normalized_gold, normalized_pred)
    stance_macro_f1 = _macro_f1(
        normalized_gold,
        normalized_pred,
        labels=("entailment", "contradiction", "neutral"),
    )

    support = total_support if total_support is not None else len(predictions)
    aligned_count = len(predictions)

    return Metrics(
        contradiction_f1=contradiction_f1,
        stance_macro_f1=stance_macro_f1,
        abstention_rate=(abstained / aligned_count if aligned_count else 0.0),
        missing_prediction_rate=(missing_prediction_count / support if support else 0.0),
        support=support,
    )
