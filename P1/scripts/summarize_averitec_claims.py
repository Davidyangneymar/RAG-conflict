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
from p1.data.averitec import averitec_record_to_claim_chunk, load_averitec_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize structured claim extraction quality on AVeriTeC claims.")
    parser.add_argument("--input", default="data/averitec/dev/dev.json")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--preview", type=int, default=6)
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> None:
    args = parse_args()
    records = load_averitec_records(args.input, limit=args.limit)
    extractor = build_claim_extractor(kind="structured", entity_backend="spacy")

    total_claims = 0
    present_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    missing_relation_examples: list[dict[str, str | None]] = []
    long_object_examples: list[dict[str, str | None]] = []

    for record in records:
        label_counts.update([record["label"]])
        claims = extractor.extract(averitec_record_to_claim_chunk(record))
        for claim in claims:
            total_claims += 1
            if claim.subject:
                present_counts["subject"] += 1
            if claim.relation:
                present_counts["relation"] += 1
                relation_counts.update([claim.relation])
            if claim.object:
                present_counts["object"] += 1
            if claim.qualifier:
                present_counts["qualifier"] += 1
            if claim.time:
                present_counts["time"] += 1
            if claim.metadata.get("structured_fields_present"):
                present_counts["structured"] += 1
            if claim.metadata.get("full_triplet_present"):
                present_counts["full_triplet"] += 1

            if not claim.relation and len(missing_relation_examples) < args.preview:
                missing_relation_examples.append(
                    {
                        "sample_id": record["sample_id"],
                        "label": record["label"],
                        "claim": record["claim"],
                        "subject": claim.subject,
                        "object": claim.object,
                        "entities": ", ".join(claim.entities),
                    }
                )
            if claim.object and len(claim.object.split()) >= 8 and len(long_object_examples) < args.preview:
                long_object_examples.append(
                    {
                        "sample_id": record["sample_id"],
                        "label": record["label"],
                        "claim": record["claim"],
                        "relation": claim.relation,
                        "object": claim.object,
                    }
                )

    payload = {
        "input": args.input,
        "evaluated_records": len(records),
        "total_claims": total_claims,
        "label_distribution": dict(label_counts),
        "fill_rates": {
            "structured_fields_present": round(safe_ratio(present_counts["structured"], total_claims), 4),
            "full_triplet_present": round(safe_ratio(present_counts["full_triplet"], total_claims), 4),
            "subject": round(safe_ratio(present_counts["subject"], total_claims), 4),
            "relation": round(safe_ratio(present_counts["relation"], total_claims), 4),
            "object": round(safe_ratio(present_counts["object"], total_claims), 4),
            "qualifier": round(safe_ratio(present_counts["qualifier"], total_claims), 4),
            "time": round(safe_ratio(present_counts["time"], total_claims), 4),
        },
        "top_relations": relation_counts.most_common(12),
        "missing_relation_examples": missing_relation_examples,
        "long_object_examples": long_object_examples,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
