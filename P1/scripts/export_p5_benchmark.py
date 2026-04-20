from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.benchmark import pipeline_output_to_benchmark_record
from p1.data.averitec import averitec_record_to_retrieval_input, load_averitec_records
from p1.data.fnc1 import read_jsonl, sample_to_retrieval_input
from p1.pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export benchmark-ready P1 outputs for P5.")
    parser.add_argument("--dataset", choices=["averitec", "fnc1"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--body-mode", default="top2_span", choices=["full", "best_sentence", "top2_span", "top3_span"])
    parser.add_argument("--max-questions", type=int, default=2)
    parser.add_argument("--max-answers-per-question", type=int, default=2)
    parser.add_argument("--output", default="")
    parser.add_argument("--preview", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = build_pipeline(extractor_kind="structured", entity_backend="spacy")

    if args.dataset == "averitec":
        records = load_averitec_records(args.input, limit=args.limit)
        retrieval_inputs = [
            averitec_record_to_retrieval_input(
                record,
                max_questions=args.max_questions,
                max_answers_per_question=args.max_answers_per_question,
            )
            for record in records
        ]
    else:
        raw_records = read_jsonl(args.input)[: args.limit]
        retrieval_inputs = [
            sample_to_retrieval_input(
                record,
                body_mode=args.body_mode,
            )
            for record in raw_records
        ]

    benchmark_records = []
    predicted_counts = Counter()
    gold_counts = Counter()
    for retrieval_input in retrieval_inputs:
        output = pipeline.run_retrieval_input(retrieval_input)
        benchmark_record = pipeline_output_to_benchmark_record(output, retrieval_input)
        benchmark_records.append(benchmark_record)
        gold_counts.update([benchmark_record["gold_label"]])
        if benchmark_record["predicted_label"]:
            predicted_counts.update([benchmark_record["predicted_label"]])

    payload = {
        "dataset": args.dataset,
        "input": args.input,
        "evaluated_records": len(benchmark_records),
        "gold_distribution": dict(gold_counts),
        "predicted_distribution": dict(predicted_counts),
        "preview": benchmark_records[: args.preview],
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in benchmark_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        payload["output"] = str(output_path)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
