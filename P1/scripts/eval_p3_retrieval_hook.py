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

from p1.data.averitec import averitec_record_to_retrieval_input, load_averitec_records
from p1.data.retrieval import read_retrieval_inputs
from p1.pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full P1 pipeline on retrieval-shaped AVeriTeC input.")
    parser.add_argument("--input", default="data/averitec/dev/dev.json")
    parser.add_argument("--input-kind", default="averitec", choices=["averitec", "retrieval_json"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--extractor-kind", default="structured", choices=["sentence", "structured"])
    parser.add_argument("--entity-backend", default="spacy")
    parser.add_argument("--max-questions", type=int, default=2)
    parser.add_argument("--max-answers-per-question", type=int, default=2)
    parser.add_argument("--preview", type=int, default=4)
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> None:
    args = parse_args()
    pipeline = build_pipeline(extractor_kind=args.extractor_kind, entity_backend=args.entity_backend)
    if args.input_kind == "averitec":
        records = load_averitec_records(args.input, limit=args.limit)
        retrieval_inputs = [
            averitec_record_to_retrieval_input(
                record,
                max_questions=args.max_questions,
                max_answers_per_question=args.max_answers_per_question,
            )
            for record in records
        ]
        labels = {item.sample_id: item.label for item in retrieval_inputs}
    else:
        retrieval_inputs = read_retrieval_inputs(args.input, limit=args.limit)
        labels = {item.sample_id: item.label for item in retrieval_inputs}

    record_count = 0
    total_input_chunks = 0
    total_claims = 0
    total_pairs = 0
    total_cross_source_pairs = 0
    records_with_retrieved_chunks = 0
    records_with_cross_source_pairs = 0
    records_with_decisive_cross_source_nli = 0
    cross_source_nli_labels: Counter[str] = Counter()
    preview_records: list[dict[str, object]] = []

    for retrieval_input in retrieval_inputs:
        output = pipeline.run_retrieval_input(retrieval_input)

        record_count += 1
        total_input_chunks += 1 + len(retrieval_input.retrieved_chunks)
        total_claims += len(output.claims)
        total_pairs += len(output.candidate_pairs)

        if retrieval_input.retrieved_chunks:
            records_with_retrieved_chunks += 1

        cross_source_pairs = []
        cross_source_labels = []
        for pair, result in zip(output.candidate_pairs, output.nli_results):
            role_a = pair.claim_a.source.metadata.get("role")
            role_b = pair.claim_b.source.metadata.get("role")
            if {role_a, role_b} == {"query", "retrieved_evidence"}:
                cross_source_pairs.append(pair)
                cross_source_labels.append(result.label.value)
                cross_source_nli_labels.update([result.label.value])

        total_cross_source_pairs += len(cross_source_pairs)
        if cross_source_pairs:
            records_with_cross_source_pairs += 1
        if any(label in {"entailment", "contradiction"} for label in cross_source_labels):
            records_with_decisive_cross_source_nli += 1

        if len(preview_records) < args.preview:
            preview_records.append(
                {
                    "sample_id": retrieval_input.sample_id,
                    "label": labels.get(retrieval_input.sample_id),
                    "retrieved_chunks": len(retrieval_input.retrieved_chunks),
                    "claims": len(output.claims),
                    "candidate_pairs": len(output.candidate_pairs),
                    "cross_source_pairs": len(cross_source_pairs),
                    "cross_source_nli_labels": Counter(cross_source_labels),
                }
            )

    payload = {
        "input": args.input,
        "evaluated_records": record_count,
        "avg_input_chunks_per_record": round(safe_ratio(total_input_chunks, record_count), 4),
        "avg_claims_per_record": round(safe_ratio(total_claims, record_count), 4),
        "avg_candidate_pairs_per_record": round(safe_ratio(total_pairs, record_count), 4),
        "avg_cross_source_pairs_per_record": round(safe_ratio(total_cross_source_pairs, record_count), 4),
        "records_with_retrieved_chunks": records_with_retrieved_chunks,
        "records_with_cross_source_pairs": records_with_cross_source_pairs,
        "records_with_decisive_cross_source_nli": records_with_decisive_cross_source_nli,
        "cross_source_pair_coverage": round(safe_ratio(records_with_cross_source_pairs, record_count), 4),
        "decisive_cross_source_nli_coverage": round(safe_ratio(records_with_decisive_cross_source_nli, record_count), 4),
        "cross_source_nli_distribution": dict(cross_source_nli_labels),
        "preview": preview_records,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
