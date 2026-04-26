from __future__ import annotations

from dataclasses import asdict

from .adapters import read_predictions
from .metrics import compute_metrics
from .schemas import PredictionRecord


def evaluate_baselines(
    gold_rows: list[dict],
    baseline_predictions: dict[str, list[dict]],
    min_align_rate: float = 0.8,
) -> dict:
    gold_by_id = {str(row["sample_id"]): row for row in gold_rows}
    gold_ids = list(gold_by_id.keys())
    record_count = len(gold_ids)

    results: dict[str, dict] = {}
    unmatched: dict[str, list[str]] = {}
    for name, rows in baseline_predictions.items():
        raw_preds = read_predictions(rows)
        pred_by_id = {p.sample_id: p for p in raw_preds}

        aligned_gold_labels: list[str] = []
        aligned_predictions: list[PredictionRecord] = []
        missing_ids: list[str] = []

        for sample_id in gold_ids:
            pred = pred_by_id.get(sample_id)
            if pred is None:
                missing_ids.append(sample_id)
                continue
            aligned_gold_labels.append(str(gold_by_id[sample_id].get("gold_label", "neutral")))
            aligned_predictions.append(pred)

        align_rate = (len(aligned_predictions) / record_count) if record_count else 0.0
        if record_count > 0 and align_rate < min_align_rate:
            raise ValueError(
                f"Baseline '{name}' alignment rate {align_rate:.4f} is below min_align_rate={min_align_rate:.4f}."
            )

        metrics = compute_metrics(
            gold_labels=aligned_gold_labels,
            predictions=aligned_predictions,
            total_support=record_count,
            missing_prediction_count=len(missing_ids),
        )
        result = asdict(metrics)
        result["alignment_rate"] = align_rate
        result["aligned_count"] = len(aligned_predictions)
        result["missing_count"] = len(missing_ids)
        results[name] = result
        unmatched[name] = missing_ids

    return {
        "record_count": record_count,
        "min_align_rate": min_align_rate,
        "baselines": results,
        "unmatched_sample_ids": unmatched,
    }
