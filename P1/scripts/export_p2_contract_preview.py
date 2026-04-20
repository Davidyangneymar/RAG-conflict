from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.data.averitec import averitec_record_to_retrieval_input, load_averitec_records
from p1.handoff import pipeline_output_to_p2_payload
from p1.pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a P2-facing preview payload from the P1 pipeline.")
    parser.add_argument("--input", default="data/averitec/dev/dev.json")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-questions", type=int, default=2)
    parser.add_argument("--max-answers-per-question", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_averitec_records(args.input, limit=args.limit)
    pipeline = build_pipeline(extractor_kind="structured", entity_backend="spacy")

    payload = []
    for record in records:
        retrieval_input = averitec_record_to_retrieval_input(
            record,
            max_questions=args.max_questions,
            max_answers_per_question=args.max_answers_per_question,
        )
        output = pipeline.run_retrieval_input(retrieval_input)
        payload.append(
            pipeline_output_to_p2_payload(
                output,
                sample_id=record["sample_id"],
            )
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
