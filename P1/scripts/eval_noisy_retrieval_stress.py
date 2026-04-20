from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.benchmark import pipeline_output_to_benchmark_record
from p1.data.averitec import averitec_record_to_retrieval_input, load_averitec_records
from p1.data.stress import build_fnc1_distractor_pool, inject_distractor_chunks
from p1.pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress-test retrieval noise by injecting unrelated distractor chunks.")
    parser.add_argument("--input", default="data/averitec/dev/dev.json")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--max-questions", type=int, default=2)
    parser.add_argument("--max-answers-per-question", type=int, default=2)
    parser.add_argument("--distractor-pool", default="data/processed/fnc1_train.jsonl")
    parser.add_argument("--distractor-pool-limit", type=int, default=500)
    parser.add_argument("--distractor-stances", default="unrelated")
    parser.add_argument("--distractor-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--preview", type=int, default=4)
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def summarize(records: list[dict]) -> dict:
    count = len(records)
    return {
        "evaluated_records": count,
        "avg_retrieved_chunk_count": round(sum(item["retrieved_chunk_count"] for item in records) / count, 4) if count else 0.0,
        "avg_claim_count": round(sum(item["claim_count"] for item in records) / count, 4) if count else 0.0,
        "avg_candidate_pair_count": round(sum(item["candidate_pair_count"] for item in records) / count, 4) if count else 0.0,
        "avg_cross_source_pair_count": round(sum(item["cross_source_pair_count"] for item in records) / count, 4) if count else 0.0,
        "cross_source_pair_coverage": round(safe_ratio(sum(1 for item in records if item["cross_source_pair_count"] > 0), count), 4),
        "decisive_prediction_coverage": round(safe_ratio(sum(1 for item in records if item["predicted_label"] in {"entailment", "contradiction"}), count), 4),
    }


def main() -> None:
    args = parse_args()
    records = load_averitec_records(args.input, limit=args.limit)
    distractor_pool = build_fnc1_distractor_pool(
        args.distractor_pool,
        limit=args.distractor_pool_limit,
        stance_labels={item.strip() for item in args.distractor_stances.split(",") if item.strip()},
    )
    pipeline = build_pipeline(extractor_kind="structured", entity_backend="spacy")

    baseline_records = []
    noisy_records = []
    preview = []

    for record in records:
        base_input = averitec_record_to_retrieval_input(
            record,
            max_questions=args.max_questions,
            max_answers_per_question=args.max_answers_per_question,
        )
        noisy_input = inject_distractor_chunks(
            base_input,
            distractor_pool,
            distractor_count=args.distractor_count,
            seed=args.seed,
        )

        baseline_output = pipeline.run_retrieval_input(base_input)
        noisy_output = pipeline.run_retrieval_input(noisy_input)

        baseline_row = pipeline_output_to_benchmark_record(baseline_output, base_input)
        noisy_row = pipeline_output_to_benchmark_record(noisy_output, noisy_input)
        baseline_records.append(baseline_row)
        noisy_records.append(noisy_row)

        if len(preview) < args.preview:
            preview.append(
                {
                    "sample_id": record["sample_id"],
                    "gold_label": record["label"],
                    "baseline": {
                        "retrieved_chunk_count": baseline_row["retrieved_chunk_count"],
                        "candidate_pair_count": baseline_row["candidate_pair_count"],
                        "cross_source_pair_count": baseline_row["cross_source_pair_count"],
                        "predicted_label": baseline_row["predicted_label"],
                    },
                    "noisy": {
                        "retrieved_chunk_count": noisy_row["retrieved_chunk_count"],
                        "candidate_pair_count": noisy_row["candidate_pair_count"],
                        "cross_source_pair_count": noisy_row["cross_source_pair_count"],
                        "predicted_label": noisy_row["predicted_label"],
                    },
                }
            )

    payload = {
        "input": args.input,
        "distractor_pool": args.distractor_pool,
        "distractor_count": args.distractor_count,
        "distractor_stances": args.distractor_stances,
        "baseline": summarize(baseline_records),
        "noisy": summarize(noisy_records),
        "delta": {
            "avg_claim_count": round(summarize(noisy_records)["avg_claim_count"] - summarize(baseline_records)["avg_claim_count"], 4),
            "avg_candidate_pair_count": round(summarize(noisy_records)["avg_candidate_pair_count"] - summarize(baseline_records)["avg_candidate_pair_count"], 4),
            "avg_cross_source_pair_count": round(summarize(noisy_records)["avg_cross_source_pair_count"] - summarize(baseline_records)["avg_cross_source_pair_count"], 4),
            "cross_source_pair_coverage": round(summarize(noisy_records)["cross_source_pair_coverage"] - summarize(baseline_records)["cross_source_pair_coverage"], 4),
            "decisive_prediction_coverage": round(summarize(noisy_records)["decisive_prediction_coverage"] - summarize(baseline_records)["decisive_prediction_coverage"], 4),
        },
        "preview": preview,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
