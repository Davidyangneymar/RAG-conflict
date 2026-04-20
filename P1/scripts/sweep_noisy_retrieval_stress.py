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
    parser = argparse.ArgumentParser(description="Sweep noisy-retrieval stress settings on AVeriTeC.")
    parser.add_argument("--input", default="data/averitec/dev/dev.json")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--max-questions", type=int, default=2)
    parser.add_argument("--max-answers-per-question", type=int, default=2)
    parser.add_argument("--distractor-pool", default="data/processed/fnc1_train.jsonl")
    parser.add_argument("--distractor-pool-limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
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
    pipeline = build_pipeline(extractor_kind="structured", entity_backend="spacy")
    retrieval_inputs = [
        averitec_record_to_retrieval_input(
            record,
            max_questions=args.max_questions,
            max_answers_per_question=args.max_answers_per_question,
        )
        for record in records
    ]

    baseline_rows = []
    for retrieval_input in retrieval_inputs:
        output = pipeline.run_retrieval_input(retrieval_input)
        baseline_rows.append(pipeline_output_to_benchmark_record(output, retrieval_input))
    baseline_summary = summarize(baseline_rows)

    configs = [
        {"name": "unrelated_x2", "stances": {"unrelated"}, "count": 2},
        {"name": "unrelated_x5", "stances": {"unrelated"}, "count": 5},
        {"name": "disagree_x2", "stances": {"disagree"}, "count": 2},
        {"name": "disagree_x5", "stances": {"disagree"}, "count": 5},
        {"name": "mixed_x5", "stances": {"unrelated", "disagree"}, "count": 5},
    ]

    results = []
    for config in configs:
        distractor_pool = build_fnc1_distractor_pool(
            args.distractor_pool,
            limit=args.distractor_pool_limit,
            stance_labels=config["stances"],
        )
        stressed_rows = []
        for retrieval_input in retrieval_inputs:
            noisy_input = inject_distractor_chunks(
                retrieval_input,
                distractor_pool,
                distractor_count=config["count"],
                seed=args.seed,
            )
            output = pipeline.run_retrieval_input(noisy_input)
            stressed_rows.append(pipeline_output_to_benchmark_record(output, noisy_input))
        stressed_summary = summarize(stressed_rows)
        results.append(
            {
                "name": config["name"],
                "distractor_stances": sorted(config["stances"]),
                "distractor_count": config["count"],
                "summary": stressed_summary,
                "delta": {
                    "avg_claim_count": round(stressed_summary["avg_claim_count"] - baseline_summary["avg_claim_count"], 4),
                    "avg_candidate_pair_count": round(stressed_summary["avg_candidate_pair_count"] - baseline_summary["avg_candidate_pair_count"], 4),
                    "avg_cross_source_pair_count": round(stressed_summary["avg_cross_source_pair_count"] - baseline_summary["avg_cross_source_pair_count"], 4),
                    "cross_source_pair_coverage": round(stressed_summary["cross_source_pair_coverage"] - baseline_summary["cross_source_pair_coverage"], 4),
                    "decisive_prediction_coverage": round(stressed_summary["decisive_prediction_coverage"] - baseline_summary["decisive_prediction_coverage"], 4),
                },
            }
        )

    payload = {
        "input": args.input,
        "baseline": baseline_summary,
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
