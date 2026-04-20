from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.blocking import BlockingConfig, MultiStageBlocker
from p1.data.ramdocs import load_ramdocs_records, ramdocs_record_to_retrieval_input
from p1.pipeline import build_pipeline


DOC_TYPES = ("correct", "misinfo", "noise")
NLI_CUE_TOKENS = {
    "no",
    "not",
    "never",
    "false",
    "deny",
    "denied",
    "without",
    "fake",
    "hoax",
    "debunked",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate P1 on real RAMDocs retrieval noise.")
    parser.add_argument("--input", default="data/RAMDocs-main/RAMDocs_test.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--preview", type=int, default=4)
    parser.add_argument("--query-threshold", type=float, default=None)
    parser.add_argument(
        "--answer-aware",
        action="store_true",
        help="Append RAMDocs answer-aware declarative evidence sentences for stress diagnosis.",
    )
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def diagnose_no_cross_source_pair(
    *,
    query_claims: list,
    evidence_claims: list,
    blocker,
) -> str:
    if not query_claims:
        return "no_query_claims"
    if not evidence_claims:
        return "no_evidence_claims"

    saw_entity_only = False
    saw_lexical_only = False
    saw_both_gate = False

    for query_claim in query_claims:
        for evidence_claim in evidence_claims:
            entity_overlap = sorted(set(query_claim.entities) & set(evidence_claim.entities))
            entity_pass = blocker._passes_entity_stage(query_claim, evidence_claim, entity_overlap)
            built_pair = blocker._build_pair(query_claim, evidence_claim)
            if built_pair is not None:
                return "unexpected_pair_present"

            lexical_similarity_value = _lexical_similarity(blocker, query_claim.text, evidence_claim.text)
            lexical_pass = blocker._passes_lexical_stage(query_claim, evidence_claim, lexical_similarity_value)

            if entity_pass and not lexical_pass:
                saw_lexical_only = True
            elif lexical_pass and not entity_pass:
                saw_entity_only = True
            elif not lexical_pass and not entity_pass:
                saw_both_gate = True

    if saw_entity_only and not saw_lexical_only and not saw_both_gate:
        return "entity_gate"
    if saw_lexical_only and not saw_entity_only and not saw_both_gate:
        return "lexical_gate"
    if saw_both_gate and not saw_entity_only and not saw_lexical_only:
        return "entity_and_lexical_gate"
    if saw_entity_only or saw_lexical_only or saw_both_gate:
        return "mixed_gate"
    return "unknown_gate"


def _lexical_similarity(blocker, left_text: str, right_text: str) -> float:
    tokenize = blocker._build_pair.__globals__["_tokenize"]
    jaccard = blocker._build_pair.__globals__["_jaccard"]
    return jaccard(tokenize(left_text), tokenize(right_text))


def diagnose_only_neutral(neutral_pairs: list[tuple]) -> str:
    if not neutral_pairs:
        return "no_neutral_pairs"

    max_lexical = max(pair.lexical_similarity for pair, _result in neutral_pairs)
    max_decisive_score = max(
        max(result.entailment_score, result.contradiction_score)
        for _pair, result in neutral_pairs
    )
    has_nli_cue = any(
        _has_nli_cue(pair.claim_a.text) or _has_nli_cue(pair.claim_b.text)
        for pair, _result in neutral_pairs
    )

    if max_lexical < 0.18:
        return "weak_lexical_signal"
    if max_lexical < 0.4 and not has_nli_cue:
        return "below_entailment_threshold_no_cue"
    if has_nli_cue and max_decisive_score < 0.5:
        return "cue_present_but_bidirectional_neutral"
    return "neutral_score_dominates"


def _has_nli_cue(text: str) -> bool:
    tokens = {token.strip(".,;:!?()[]{}\"'").lower() for token in text.split()}
    return bool(tokens & NLI_CUE_TOKENS)


def main() -> None:
    args = parse_args()
    records = load_ramdocs_records(args.input, limit=args.limit)
    blocker = MultiStageBlocker(
        config=BlockingConfig(query_pair_min_lexical_similarity=args.query_threshold)
    )
    pipeline = build_pipeline(extractor_kind="structured", entity_backend="spacy", blocker=blocker)

    total_docs = Counter()
    retained_docs = Counter()
    decisive_docs = Counter()
    failure_buckets_by_doc_type = {doc_type: Counter() for doc_type in DOC_TYPES}
    no_pair_subreasons_by_doc_type = {doc_type: Counter() for doc_type in DOC_TYPES}
    only_neutral_subreasons_by_doc_type = {doc_type: Counter() for doc_type in DOC_TYPES}
    label_by_doc_type = {doc_type: Counter() for doc_type in DOC_TYPES}
    records_with_type = Counter()
    records_with_retained_type = Counter()
    records_with_decisive_type = Counter()
    preview = []

    for record in records:
        retrieval_input = ramdocs_record_to_retrieval_input(record, answer_aware=args.answer_aware)
        output = pipeline.run_retrieval_input(retrieval_input)

        doc_type_by_chunk = {}
        for chunk in retrieval_input.retrieved_chunks:
            doc_type = chunk.metadata.get("ramdocs_type", "unknown")
            doc_type_by_chunk[chunk.chunk_id] = doc_type
            total_docs.update([doc_type])
        present_types = set(doc_type_by_chunk.values())
        for doc_type in present_types:
            records_with_type.update([doc_type])

        query_claims = [
            claim
            for claim in output.claims
            if str(claim.source.metadata.get("role") or "").strip().lower() == "query"
        ]
        evidence_claims_by_chunk = defaultdict(list)
        for claim in output.claims:
            role = str(claim.source.metadata.get("role") or "").strip().lower()
            if role != "retrieved_evidence":
                continue
            evidence_claims_by_chunk[claim.source_chunk_id].append(claim)

        chunk_labels = defaultdict(list)
        neutral_pairs_by_chunk = defaultdict(list)
        for pair, result in zip(output.candidate_pairs, output.nli_results):
            role_a = pair.claim_a.source.metadata.get("role")
            role_b = pair.claim_b.source.metadata.get("role")
            if {role_a, role_b} != {"query", "retrieved_evidence"}:
                continue
            evidence_claim = pair.claim_a if role_a == "retrieved_evidence" else pair.claim_b
            chunk_id = evidence_claim.source_chunk_id
            doc_type = doc_type_by_chunk.get(chunk_id)
            if not doc_type:
                continue
            label_by_doc_type[doc_type].update([result.label.value])
            chunk_labels[chunk_id].append(result.label.value)
            if result.label.value == "neutral":
                neutral_pairs_by_chunk[chunk_id].append((pair, result))

        retained_chunk_ids = set(chunk_labels.keys())
        decisive_chunk_ids = {
            chunk_id
            for chunk_id, labels in chunk_labels.items()
            if any(label in {"entailment", "contradiction"} for label in labels)
        }
        diagnosis_by_chunk = {}
        for chunk in retrieval_input.retrieved_chunks:
            chunk_id = chunk.chunk_id
            if not query_claims:
                diagnosis = "no_query_claims"
            elif not evidence_claims_by_chunk.get(chunk_id):
                diagnosis = "no_evidence_claims"
            elif chunk_id not in retained_chunk_ids:
                diagnosis = "no_cross_source_pair"
            elif chunk_id not in decisive_chunk_ids:
                diagnosis = "only_neutral"
            else:
                diagnosis = "decisive"
            diagnosis_by_chunk[chunk_id] = diagnosis
            doc_type = doc_type_by_chunk[chunk_id]
            failure_buckets_by_doc_type[doc_type].update([diagnosis])
            if diagnosis == "no_cross_source_pair":
                no_pair_subreason = diagnose_no_cross_source_pair(
                    query_claims=query_claims,
                    evidence_claims=evidence_claims_by_chunk.get(chunk_id, []),
                    blocker=pipeline.blocker,
                )
                no_pair_subreasons_by_doc_type[doc_type].update([no_pair_subreason])
            elif diagnosis == "only_neutral":
                only_neutral_subreason = diagnose_only_neutral(neutral_pairs_by_chunk.get(chunk_id, []))
                only_neutral_subreasons_by_doc_type[doc_type].update([only_neutral_subreason])

        retained_types = set()
        decisive_types = set()
        for chunk_id in retained_chunk_ids:
            doc_type = doc_type_by_chunk[chunk_id]
            retained_docs.update([doc_type])
            retained_types.add(doc_type)
        for chunk_id in decisive_chunk_ids:
            doc_type = doc_type_by_chunk[chunk_id]
            decisive_docs.update([doc_type])
            decisive_types.add(doc_type)
        for doc_type in retained_types:
            records_with_retained_type.update([doc_type])
        for doc_type in decisive_types:
            records_with_decisive_type.update([doc_type])

        if len(preview) < args.preview:
            preview.append(
                {
                    "sample_id": record["sample_id"],
                    "question": record["question"],
                    "query_claim_count": len(query_claims),
                    "doc_type_counts": dict(Counter(doc["type"] for doc in record["documents"])),
                    "diagnosis_by_chunk": {
                        chunk["doc_id"]: diagnosis_by_chunk.get(chunk["doc_id"])
                        for chunk in record["documents"]
                    },
                    "retained_doc_types": sorted(retained_types),
                    "decisive_doc_types": sorted(decisive_types),
                }
            )

    payload = {
        "input": args.input,
        "evaluated_records": len(records),
        "query_threshold": args.query_threshold,
        "answer_aware": args.answer_aware,
        "document_counts": dict(total_docs),
        "retained_document_counts": dict(retained_docs),
        "decisive_document_counts": dict(decisive_docs),
        "retained_doc_coverage_by_type": {
            doc_type: round(safe_ratio(retained_docs[doc_type], total_docs[doc_type]), 4)
            for doc_type in DOC_TYPES
        },
        "decisive_doc_coverage_by_type": {
            doc_type: round(safe_ratio(decisive_docs[doc_type], total_docs[doc_type]), 4)
            for doc_type in DOC_TYPES
        },
        "record_level_retained_coverage_by_type": {
            doc_type: round(safe_ratio(records_with_retained_type[doc_type], records_with_type[doc_type]), 4)
            for doc_type in DOC_TYPES
        },
        "record_level_decisive_coverage_by_type": {
            doc_type: round(safe_ratio(records_with_decisive_type[doc_type], records_with_type[doc_type]), 4)
            for doc_type in DOC_TYPES
        },
        "nli_label_distribution_by_doc_type": {
            doc_type: dict(label_by_doc_type[doc_type])
            for doc_type in DOC_TYPES
        },
        "failure_buckets_by_doc_type": {
            doc_type: dict(failure_buckets_by_doc_type[doc_type])
            for doc_type in DOC_TYPES
        },
        "no_cross_source_pair_subreasons_by_doc_type": {
            doc_type: dict(no_pair_subreasons_by_doc_type[doc_type])
            for doc_type in DOC_TYPES
        },
        "only_neutral_subreasons_by_doc_type": {
            doc_type: dict(only_neutral_subreasons_by_doc_type[doc_type])
            for doc_type in DOC_TYPES
        },
        "preview": preview,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
