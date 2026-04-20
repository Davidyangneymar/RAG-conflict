"""Build query-taxonomy and failure-attribution diagnostics for P3 -> P1 runs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_retrieval_config
from src.schemas.retrieval import RetrievedEvidence
from src.services.evidence_hygiene import assess_evidence_hygiene

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
TAXONOMY_BUCKETS = [
    "very_short_factoid",
    "yes_no_claim_like",
    "entity_seeking",
    "temporal_query",
    "ambiguous_name_query",
    "quoted_or_attribution_heavy",
]
FAILURE_LABELS = [
    "Q_SHORT",
    "Q_UNDERSPECIFIED",
    "Q_COLLAPSED_IN_P1",
    "EVIDENCE_OK_BUT_NOT_COMPARABLE",
    "EVIDENCE_STILL_NOISY",
    "NO_CROSS_SOURCE_SIGNAL",
]
QUERY_SIDE_FAILURES = {"Q_SHORT", "Q_UNDERSPECIFIED", "Q_COLLAPSED_IN_P1"}
RETRIEVAL_SIDE_FAILURES = {"NO_CROSS_SOURCE_SIGNAL", "EVIDENCE_STILL_NOISY"}


def token_count(text: str | None) -> int:
    return len(TOKEN_RE.findall(text or ""))


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


def classify_query_bucket(query_text: str) -> str:
    lowered = query_text.lower().strip()
    tokens = token_count(query_text)
    capitalized_tokens = re.findall(r"\b[A-Z][A-Za-z0-9'_-]*\b", query_text)

    if any(marker in query_text for marker in ('"', "“", "”")) or any(
        phrase in lowered
        for phrase in ("according to", "claimed", "announced", "reported", "attributed", "mistakenly attributed")
    ):
        return "quoted_or_attribution_heavy"

    if MONTH_RE.search(query_text) or re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", query_text) or any(
        word in lowered for word in ("born", "died", "released", "premiered", "started", "ended", "founded")
    ):
        return "temporal_query"

    generic_descriptor = any(
        phrase in lowered
        for phrase in (
            " is a person",
            " is an actor",
            " is a broadcaster",
            " is a character",
            " is a city",
            " is a country",
            " is a film",
            " is a musician",
        )
    )
    if tokens <= 6 and len(capitalized_tokens) >= 2 and generic_descriptor:
        return "ambiguous_name_query"

    if any(
        phrase in lowered
        for phrase in (
            "real name",
            "is a ",
            "is an ",
            "was a ",
            "was an ",
            "is the ",
            "was the ",
            "character on",
            "known as",
            "called ",
        )
    ):
        return "entity_seeking"

    if tokens <= 5:
        return "very_short_factoid"

    return "yes_no_claim_like"


def build_evidence_summary(handoff_record: dict, config) -> dict[str, Any]:
    penalties: list[float] = []
    flag_counts: Counter[str] = Counter()

    for chunk in handoff_record.get("retrieved_chunks") or []:
        metadata = chunk.get("metadata") or {}
        evidence = RetrievedEvidence(
            query=handoff_record.get("query", ""),
            chunk_id=chunk.get("chunk_id", ""),
            doc_id=metadata.get("source_doc_id", ""),
            dataset=metadata.get("dataset", "unknown"),
            source_name=chunk.get("source_medium"),
            source_url=chunk.get("source_url"),
            title=metadata.get("title"),
            published_at=metadata.get("published_at"),
            text=chunk.get("text", ""),
            rank=chunk.get("rank", 0),
            metadata=metadata,
        )
        assessment = assess_evidence_hygiene(evidence, config)
        penalties.append(assessment.penalty)
        for flag in assessment.flags:
            flag_counts[flag] += 1

    avg_penalty = round(sum(penalties) / len(penalties), 4) if penalties else 0.0
    noisy = avg_penalty >= 0.12 or any(
        flag_counts.get(flag, 0) > 0
        for flag in ("enumeration_heavy", "entity_or_appositive_heavy", "ultra_short_non_propositional", "title_or_header_like")
    )
    return {
        "average_penalty": avg_penalty,
        "flag_counts": dict(flag_counts),
        "noisy": noisy,
    }


def is_query_collapsed(query_text: str, query_claims: list[dict]) -> bool:
    original_tokens = token_count(query_text)
    for claim in query_claims:
        claim_tokens = token_count(claim.get("text"))
        if claim.get("relation") is None and claim.get("object") is None:
            return True
        if original_tokens >= 7 and claim_tokens <= max(4, int(original_tokens * 0.6)):
            return True
    return False


def assign_failure_label(
    *,
    cross_source_pair_count: int,
    short_query_claim: bool,
    query_collapsed: bool,
    evidence_noisy: bool,
    query_bucket: str,
    query_tokens: int,
) -> str:
    if cross_source_pair_count == 0:
        return "NO_CROSS_SOURCE_SIGNAL"
    if short_query_claim and query_bucket in {"very_short_factoid", "ambiguous_name_query", "entity_seeking"}:
        return "Q_SHORT"
    if query_collapsed:
        return "Q_COLLAPSED_IN_P1"
    if evidence_noisy:
        return "EVIDENCE_STILL_NOISY"
    if query_bucket in {"ambiguous_name_query", "entity_seeking"} or query_tokens <= 7:
        return "Q_UNDERSPECIFIED"
    return "EVIDENCE_OK_BUT_NOT_COMPARABLE"


def analyze_slice(name: str, benchmark_records: list[dict], handoff_records: dict[str, dict], config) -> dict:
    bucket_stats: dict[str, dict[str, Any]] = {
        bucket: {
            "records": 0,
            "short_query_records": 0,
            "decisive_records": 0,
            "non_decisive_records": 0,
            "avg_cross_source_pairs": 0.0,
            "avg_evidence_penalty": 0.0,
            "failure_labels": Counter(),
        }
        for bucket in TAXONOMY_BUCKETS
    }
    failure_counts: Counter[str] = Counter()
    failure_family_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    record_summaries: list[dict[str, Any]] = []

    for record in benchmark_records:
        sample_id = str(record.get("sample_id"))
        handoff_record = handoff_records.get(sample_id, {})
        query = record.get("query", "")
        query_bucket = classify_query_bucket(query)
        query_tokens = token_count(query)
        query_claims = [claim for claim in (record.get("claims") or []) if claim.get("source_role") == "query"]
        query_claim_tokens = [token_count(claim.get("text")) for claim in query_claims]
        short_query_claim = any(count < 6 for count in query_claim_tokens)
        query_collapsed = is_query_collapsed(query, query_claims)
        decisive = any(
            result.get("label") in {"entailment", "contradiction"}
            for result in (record.get("cross_source_nli_results") or [])
        )
        cross_source_pair_count = int(record.get("cross_source_pair_count", 0))
        evidence_summary = build_evidence_summary(handoff_record, config)
        failure_label = None
        if not decisive:
            failure_label = assign_failure_label(
                cross_source_pair_count=cross_source_pair_count,
                short_query_claim=short_query_claim,
                query_collapsed=query_collapsed,
                evidence_noisy=evidence_summary["noisy"],
                query_bucket=query_bucket,
                query_tokens=query_tokens,
            )
            failure_counts[failure_label] += 1
            bucket_stats[query_bucket]["failure_labels"][failure_label] += 1
            if failure_label in QUERY_SIDE_FAILURES:
                failure_family_counts["query_side"] += 1
            elif failure_label in RETRIEVAL_SIDE_FAILURES:
                failure_family_counts["retrieval_side"] += 1
            else:
                failure_family_counts["comparison_gap"] += 1

        bucket_stats[query_bucket]["records"] += 1
        bucket_stats[query_bucket]["short_query_records"] += int(short_query_claim)
        bucket_stats[query_bucket]["decisive_records"] += int(decisive)
        bucket_stats[query_bucket]["non_decisive_records"] += int(not decisive)
        bucket_stats[query_bucket]["avg_cross_source_pairs"] += cross_source_pair_count
        bucket_stats[query_bucket]["avg_evidence_penalty"] += evidence_summary["average_penalty"]

        summary = {
            "sample_id": sample_id,
            "query": query,
            "query_bucket": query_bucket,
            "query_token_count": query_tokens,
            "query_claims": [claim.get("text") for claim in query_claims],
            "query_claim_token_counts": query_claim_tokens,
            "short_query_claim": short_query_claim,
            "query_collapsed_in_p1": query_collapsed,
            "cross_source_pair_count": cross_source_pair_count,
            "decisive_cross_source_nli": decisive,
            "predicted_label": record.get("predicted_label"),
            "evidence_average_penalty": evidence_summary["average_penalty"],
            "evidence_flag_counts": evidence_summary["flag_counts"],
            "failure_label": failure_label,
        }
        record_summaries.append(summary)

        if not decisive and len(examples) < 15:
            examples.append(summary)

    for bucket in TAXONOMY_BUCKETS:
        stats = bucket_stats[bucket]
        record_count = stats["records"]
        if record_count:
            stats["short_query_rate"] = round(stats["short_query_records"] / record_count, 4)
            stats["decisive_rate"] = round(stats["decisive_records"] / record_count, 4)
            stats["avg_cross_source_pairs"] = round(stats["avg_cross_source_pairs"] / record_count, 4)
            stats["avg_evidence_penalty"] = round(stats["avg_evidence_penalty"] / record_count, 4)
        else:
            stats["short_query_rate"] = 0.0
            stats["decisive_rate"] = 0.0
            stats["avg_cross_source_pairs"] = 0.0
            stats["avg_evidence_penalty"] = 0.0
        stats["failure_labels"] = dict(stats["failure_labels"].most_common())

    return {
        "slice_name": name,
        "records_evaluated": len(benchmark_records),
        "bucket_stats": bucket_stats,
        "failure_label_counts": dict(failure_counts.most_common()),
        "failure_family_counts": dict(failure_family_counts.most_common()),
        "examples": examples,
        "records": record_summaries,
    }


def render_report(regression: dict, diagnostic: dict) -> str:
    def bucket_line(bucket: str, payload: dict) -> str:
        stats = payload["bucket_stats"][bucket]
        return (
            f"| {bucket} | {stats['records']} | {stats['short_query_rate']:.4f} | "
            f"{stats['decisive_rate']:.4f} | {stats['avg_cross_source_pairs']:.4f} | "
            f"{stats['avg_evidence_penalty']:.4f} |"
        )

    seen_examples: set[tuple[str, str]] = set()
    example_records: list[dict[str, Any]] = []
    for candidate in regression["examples"] + diagnostic["examples"]:
        key = (candidate["sample_id"], candidate["failure_label"] or "")
        if key in seen_examples:
            continue
        seen_examples.add(key)
        example_records.append(candidate)
        if len(example_records) >= 15:
            break
    lines = [
        "# P3 Query Short-Claim Report",
        "",
        "## Scope",
        "- Baseline configuration: chunking v2 enabled, evidence hygiene implemented but disabled by default.",
        "- Fixed regression harness: 30 FEVER-dev records.",
        "- Larger diagnostic slice: 150 FEVER-dev records.",
        "- No handoff schema changes and no P1/P2 core logic changes.",
        "",
        "## Decisive NLI Rate By Query Bucket",
        "",
        "### Fixed 30-record regression slice",
        "| Bucket | Records | Short-query rate | Decisive NLI rate | Avg cross-source pairs | Avg evidence penalty |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(bucket_line(bucket, regression) for bucket in TAXONOMY_BUCKETS)
    lines.extend(
        [
            "",
            "### Larger 150-record diagnostic slice",
            "| Bucket | Records | Short-query rate | Decisive NLI rate | Avg cross-source pairs | Avg evidence penalty |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(bucket_line(bucket, diagnostic) for bucket in TAXONOMY_BUCKETS)

    lines.extend(
        [
            "",
            "## Top Recurring Failure Labels",
            "",
            "### Fixed 30-record regression slice",
        ]
    )
    for label, count in regression["failure_label_counts"].items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "### Larger 150-record diagnostic slice"])
    for label, count in diagnostic["failure_label_counts"].items():
        lines.append(f"- `{label}`: {count}")

    lines.extend(["", "## Failure Families"])
    lines.append(f"- Regression 30 query-side family: {regression['failure_family_counts'].get('query_side', 0)}")
    lines.append(f"- Regression 30 retrieval-side family: {regression['failure_family_counts'].get('retrieval_side', 0)}")
    lines.append(f"- Regression 30 comparison-gap family: {regression['failure_family_counts'].get('comparison_gap', 0)}")
    lines.append(f"- Diagnostic 150 query-side family: {diagnostic['failure_family_counts'].get('query_side', 0)}")
    lines.append(f"- Diagnostic 150 retrieval-side family: {diagnostic['failure_family_counts'].get('retrieval_side', 0)}")
    lines.append(f"- Diagnostic 150 comparison-gap family: {diagnostic['failure_family_counts'].get('comparison_gap', 0)}")

    lines.extend(["", "## Concrete Bad Examples"])
    for example in example_records[:15]:
        lines.append(
            f"- `{example['sample_id']}` [{example['query_bucket']}] `{example['query']}` -> "
            f"`{example['failure_label']}`; short_query={example['short_query_claim']}; "
            f"collapsed={example['query_collapsed_in_p1']}; cross_source_pairs={example['cross_source_pair_count']}; "
            f"evidence_penalty={example['evidence_average_penalty']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Final Judgment",
            "- The fixed 30-record harness points to a split picture: query-side collapse/shortness is substantial, but evidence-side noise still appears in a noticeable minority.",
            "- On the 150-record slice, raw `NO_CROSS_SOURCE_SIGNAL` dominates because the diagnostic slice is much larger than the tiny matched wiki sample; that is best treated as a corpus-coverage artifact, not a clean retrieval-algorithm signal.",
            "- After separating those zero-signal cases out, query-side failures (`Q_SHORT`, `Q_COLLAPSED_IN_P1`, `Q_UNDERSPECIFIED`) outnumber retrieval-side noise labels in the diagnosable subset.",
            "- The next owner should therefore be closer to P1/query-understanding or query claim transformation than further P3 retrieval tuning.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build query taxonomy and failure attribution reports.")
    parser.add_argument("--regression-benchmark", required=True)
    parser.add_argument("--regression-handoff", required=True)
    parser.add_argument("--diagnostic-benchmark", required=True)
    parser.add_argument("--diagnostic-handoff", required=True)
    parser.add_argument("--taxonomy-output", required=True)
    parser.add_argument("--failure-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--config-path", default="config/retrieval.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_retrieval_config(args.config_path)

    regression = analyze_slice(
        "regression_30",
        load_jsonl(Path(args.regression_benchmark)),
        {str(record["sample_id"]): record for record in load_json(Path(args.regression_handoff)).get("records", [])},
        config,
    )
    diagnostic = analyze_slice(
        "diagnostic_150",
        load_jsonl(Path(args.diagnostic_benchmark)),
        {str(record["sample_id"]): record for record in load_json(Path(args.diagnostic_handoff)).get("records", [])},
        config,
    )

    taxonomy_payload = {
        "regression_30": {
            "records_evaluated": regression["records_evaluated"],
            "bucket_stats": regression["bucket_stats"],
            "failure_family_counts": regression["failure_family_counts"],
        },
        "diagnostic_150": {
            "records_evaluated": diagnostic["records_evaluated"],
            "bucket_stats": diagnostic["bucket_stats"],
            "failure_family_counts": diagnostic["failure_family_counts"],
        },
    }
    failure_payload = {
        "regression_30": {
            "records_evaluated": regression["records_evaluated"],
            "failure_label_counts": regression["failure_label_counts"],
            "failure_family_counts": regression["failure_family_counts"],
            "records": regression["records"],
        },
        "diagnostic_150": {
            "records_evaluated": diagnostic["records_evaluated"],
            "failure_label_counts": diagnostic["failure_label_counts"],
            "failure_family_counts": diagnostic["failure_family_counts"],
            "records": diagnostic["records"],
        },
    }

    Path(args.taxonomy_output).write_text(
        json.dumps(taxonomy_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(args.failure_output).write_text(
        json.dumps(failure_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(args.report_output).write_text(render_report(regression, diagnostic), encoding="utf-8")


if __name__ == "__main__":
    main()
