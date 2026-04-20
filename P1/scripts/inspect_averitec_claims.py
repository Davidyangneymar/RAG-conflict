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

from p1.claim_extraction import build_claim_extractor
from p1.data.averitec import averitec_record_to_chunks, load_averitec_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview AVeriTeC claim extraction on real dataset records.")
    parser.add_argument("--input", default="data/averitec/dev/dev.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--extractor-kind", default="structured", choices=["sentence", "structured"])
    parser.add_argument("--entity-backend", default="spacy")
    parser.add_argument("--include-questions", action="store_true")
    parser.add_argument("--include-answers", action="store_true")
    parser.add_argument("--max-questions", type=int, default=1)
    parser.add_argument("--max-answers-per-question", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_averitec_records(args.input, limit=args.limit)
    extractor = build_claim_extractor(kind=args.extractor_kind, entity_backend=args.entity_backend)
    label_counts = Counter(record["label"] for record in records)

    payload: dict[str, object] = {
        "input": args.input,
        "evaluated_records": len(records),
        "labels": dict(label_counts),
        "preview": [],
    }

    for record in records:
        chunks = averitec_record_to_chunks(
            record,
            include_questions=args.include_questions,
            include_answers=args.include_answers,
            max_questions=args.max_questions,
            max_answers_per_question=args.max_answers_per_question,
        )
        preview_item = {
            "sample_id": record["sample_id"],
            "label": record["label"],
            "claim": record["claim"],
            "question_count": record["question_count"],
            "answer_count": record["answer_count"],
            "chunks": [],
        }
        for chunk in chunks:
            claims = extractor.extract(chunk)
            preview_item["chunks"].append(
                {
                    "chunk_id": chunk.chunk_id,
                    "role": chunk.metadata.get("role", "claim"),
                    "text": chunk.text,
                    "claims": [
                        {
                            "claim_id": claim.claim_id,
                            "text": claim.text,
                            "subject": claim.subject,
                            "relation": claim.relation,
                            "object": claim.object,
                            "qualifier": claim.qualifier,
                            "time": claim.time,
                            "entities": claim.entities,
                        }
                        for claim in claims
                    ],
                }
            )
        payload["preview"].append(preview_item)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
