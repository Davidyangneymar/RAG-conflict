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

from p1.blocking import BlockingConfig, MultiStageBlocker
from p1.data.ramdocs import load_ramdocs_records, ramdocs_record_to_retrieval_input
from p1.pipeline import build_pipeline


DOC_TYPES = ("correct", "misinfo", "noise")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune lexical blocking threshold on RAMDocs.")
    parser.add_argument("--input", default="data/RAMDocs-main/RAMDocs_test.jsonl")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--thresholds", default="none,0.0,0.05,0.1,0.15,0.2")
    parser.add_argument("--query-threshold", default=None)
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def evaluate_threshold(
    records: list[dict],
    lexical_threshold: float | None,
    *,
    query_lexical_threshold: float | None,
) -> dict:
    blocker = MultiStageBlocker(
        config=BlockingConfig(
            min_entity_overlap=1,
            min_lexical_similarity=lexical_threshold,
            query_pair_min_lexical_similarity=query_lexical_threshold,
            min_embedding_similarity=None,
            allow_empty_entities=True,
            combine_mode="cascade",
        )
    )
    pipeline = build_pipeline(extractor_kind="structured", entity_backend="spacy", blocker=blocker)

    total_docs = Counter()
    retained_docs = Counter()
    decisive_docs = Counter()

    for record in records:
        retrieval_input = ramdocs_record_to_retrieval_input(record)
        output = pipeline.run_retrieval_input(retrieval_input)

        doc_type_by_chunk = {}
        for chunk in retrieval_input.retrieved_chunks:
            doc_type = chunk.metadata.get("ramdocs_type", "unknown")
            doc_type_by_chunk[chunk.chunk_id] = doc_type
            total_docs.update([doc_type])

        retained_chunk_ids = set()
        decisive_chunk_ids = set()
        for pair, result in zip(output.candidate_pairs, output.nli_results):
            role_a = pair.claim_a.source.metadata.get("role")
            role_b = pair.claim_b.source.metadata.get("role")
            if {role_a, role_b} != {"query", "retrieved_evidence"}:
                continue
            evidence_claim = pair.claim_a if role_a == "retrieved_evidence" else pair.claim_b
            chunk_id = evidence_claim.source_chunk_id
            if chunk_id not in doc_type_by_chunk:
                continue
            retained_chunk_ids.add(chunk_id)
            if result.label.value in {"entailment", "contradiction"}:
                decisive_chunk_ids.add(chunk_id)

        for chunk_id in retained_chunk_ids:
            retained_docs.update([doc_type_by_chunk[chunk_id]])
        for chunk_id in decisive_chunk_ids:
            decisive_docs.update([doc_type_by_chunk[chunk_id]])

    return {
        "threshold": lexical_threshold,
        "query_threshold": query_lexical_threshold,
        "retained_doc_coverage_by_type": {
            doc_type: round(safe_ratio(retained_docs[doc_type], total_docs[doc_type]), 4)
            for doc_type in DOC_TYPES
        },
        "decisive_doc_coverage_by_type": {
            doc_type: round(safe_ratio(decisive_docs[doc_type], total_docs[doc_type]), 4)
            for doc_type in DOC_TYPES
        },
    }


def main() -> None:
    args = parse_args()
    records = load_ramdocs_records(args.input, limit=args.limit)
    thresholds: list[float | None] = []
    for raw in args.thresholds.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item == "none":
            thresholds.append(None)
        else:
            thresholds.append(float(item))

    query_threshold = None if args.query_threshold is None else float(args.query_threshold)

    payload = {
        "input": args.input,
        "evaluated_records": len(records),
        "results": [
            evaluate_threshold(records, threshold, query_lexical_threshold=query_threshold)
            for threshold in thresholds
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
