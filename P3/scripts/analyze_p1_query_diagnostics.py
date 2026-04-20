"""Diagnose short query claims and their relationship to evidence hygiene."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def classify_query_pattern(query_text: str) -> list[str]:
    lowered = query_text.lower()
    patterns: list[str] = []
    if re.search(r"\b(is|was|are|were)\b", lowered):
        patterns.append("copular_fact")
    if re.search(r"\b\d{4}\b", lowered) or re.search(r"\b\d+\b", lowered):
        patterns.append("contains_year_or_number")
    if token_count(query_text) <= 6:
        patterns.append("very_short_query")
    if re.search(r"\b(in|at|on)\b", lowered):
        patterns.append("location_or_time_phrase")
    if not patterns:
        patterns.append("other")
    return patterns


def analyze_records(benchmark_records: list[dict], handoff_records: dict[str, dict]) -> dict:
    detailed_records: list[dict] = []
    short_query_records = 0
    short_query_with_poor_evidence = 0
    decisive_short_query_records = 0
    decisive_non_short_query_records = 0
    poor_evidence_records = 0
    pattern_counter: Counter[str] = Counter()

    for record in benchmark_records:
        sample_id = str(record.get("sample_id"))
        claims = record.get("claims") or []
        query_claims = [claim for claim in claims if claim.get("source_role") == "query"]
        query_token_counts = [token_count(claim.get("text") or "") for claim in query_claims]
        short_query = any(count < 6 for count in query_token_counts)

        handoff_record = handoff_records.get(sample_id, {})
        retrieved_chunks = handoff_record.get("retrieved_chunks") or []
        penalties = []
        flags: Counter[str] = Counter()
        for chunk in retrieved_chunks:
            metadata = chunk.get("metadata") or {}
            penalties.append(float(metadata.get("hygiene_penalty", 0.0)))
            for flag in metadata.get("hygiene_flags") or []:
                flags[flag] += 1

        avg_penalty = round(sum(penalties) / len(penalties), 4) if penalties else 0.0
        poor_evidence = avg_penalty >= 0.18 or any(
            flag in {"enumeration_heavy", "parenthetical_heavy", "entity_or_appositive_heavy"}
            for flag in flags
        )
        decisive = any(
            result.get("label") in {"entailment", "contradiction"}
            for result in (record.get("cross_source_nli_results") or [])
        )

        if poor_evidence:
            poor_evidence_records += 1
        if short_query:
            short_query_records += 1
            if poor_evidence:
                short_query_with_poor_evidence += 1
            if decisive:
                decisive_short_query_records += 1
            for claim in query_claims:
                for pattern in classify_query_pattern(claim.get("text") or ""):
                    pattern_counter[pattern] += 1
        elif decisive:
            decisive_non_short_query_records += 1

        detailed_records.append(
            {
                "sample_id": sample_id,
                "query": record.get("query"),
                "query_claims": [claim.get("text") for claim in query_claims],
                "query_claim_token_counts": query_token_counts,
                "short_query_claim": short_query,
                "average_evidence_hygiene_penalty": avg_penalty,
                "evidence_hygiene_flags": dict(flags),
                "cross_source_pair_count": record.get("cross_source_pair_count", 0),
                "decisive_cross_source_nli": decisive,
                "predicted_label": record.get("predicted_label"),
            }
        )

    short_query_failures = [
        item
        for item in detailed_records
        if item["short_query_claim"] and not item["decisive_cross_source_nli"]
    ]
    short_query_failures.sort(
        key=lambda item: (
            -item["average_evidence_hygiene_penalty"],
            item["cross_source_pair_count"],
            item["sample_id"],
        )
    )

    return {
        "records_evaluated": len(benchmark_records),
        "records_with_short_query_claims": short_query_records,
        "records_with_short_query_claims_ratio": round(short_query_records / len(benchmark_records), 4)
        if benchmark_records
        else 0.0,
        "records_with_poor_evidence": poor_evidence_records,
        "records_with_poor_evidence_ratio": round(poor_evidence_records / len(benchmark_records), 4)
        if benchmark_records
        else 0.0,
        "short_query_and_poor_evidence_records": short_query_with_poor_evidence,
        "decisive_short_query_records": decisive_short_query_records,
        "decisive_non_short_query_records": decisive_non_short_query_records,
        "dominant_short_query_patterns": pattern_counter.most_common(5),
        "example_short_query_failures": short_query_failures[:10],
        "records": detailed_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze query-side short claims against evidence hygiene.")
    parser.add_argument("--benchmark-input", required=True, help="Path to P1 benchmark export JSONL.")
    parser.add_argument("--handoff-input", required=True, help="Path to P3 handoff batch JSON.")
    parser.add_argument("--output", required=True, help="Path to write diagnostics JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_records = load_jsonl(Path(args.benchmark_input))
    handoff_payload = load_json(Path(args.handoff_input))
    handoff_records = {str(record["sample_id"]): record for record in handoff_payload.get("records", [])}
    payload = analyze_records(benchmark_records, handoff_records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
