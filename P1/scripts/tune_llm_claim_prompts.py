from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.claim_extraction import build_claim_extractor
from p1.data.averitec import averitec_record_to_claim_chunk, load_averitec_records
from p1.data.fnc1 import read_jsonl
from p1.schemas import ChunkInput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM prompt variants for structured claim extraction.")
    parser.add_argument("--dataset", choices=["fnc1", "averitec"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--preview", type=int, default=4)
    parser.add_argument("--entity-backend", choices=["auto", "regex", "spacy"], default="spacy")
    parser.add_argument("--prompt-variants", default="baseline,headline_aware")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-style", choices=["chat_completions", "responses"], default="chat_completions")
    parser.add_argument("--llm-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--llm-batch-size", type=int, default=1)
    parser.add_argument("--fallback-to-heuristic", action="store_true")
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def load_chunks(dataset: str, input_path: str, offset: int, limit: int) -> list[tuple[str, ChunkInput]]:
    if dataset == "fnc1":
        records = read_jsonl(input_path)[offset : offset + limit]
        return [
            (
                sample["sample_id"],
                ChunkInput(
                    doc_id=sample["sample_id"],
                    chunk_id="headline",
                    text=sample["headline"],
                    metadata={"stance_label": sample["stance_label"], "dataset": "fnc1"},
                ),
            )
            for sample in records
        ]
    records = load_averitec_records(input_path, limit=offset + limit)[offset:]
    return [(record["sample_id"], averitec_record_to_claim_chunk(record)) for record in records]


def summarize_variant(
    *,
    chunks: list[tuple[str, ChunkInput]],
    entity_backend: str,
    prompt_variant: str,
    llm_model: str | None,
    llm_base_url: str | None,
    llm_api_style: str,
    llm_timeout_seconds: float,
    llm_batch_size: int,
    fallback_to_heuristic: bool,
    preview: int,
) -> dict[str, object]:
    extractor = build_claim_extractor(
        kind="structured_llm",
        entity_backend=entity_backend,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_style=llm_api_style,
        prompt_variant=prompt_variant,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_batch_size=llm_batch_size,
        fallback_to_heuristic=fallback_to_heuristic,
    )

    present_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    success_claims = 0
    fallback_claims = 0
    cache_hit_claims = 0
    total_claims = 0
    preview_examples: list[dict[str, object]] = []
    started_at = time.perf_counter()

    all_claims = extractor.extract_many([chunk for _, chunk in chunks])
    chunk_lookup: dict[str, str] = {}
    for sample_id, chunk in chunks:
        chunk_lookup[f"{chunk.doc_id}:{chunk.chunk_id or 'chunk0'}"] = sample_id

    for claim in all_claims:
        sample_id = chunk_lookup.get(f"{claim.source_doc_id}:{claim.source_chunk_id or 'chunk0'}", claim.source_doc_id or "")
        total_claims += 1
        if claim.metadata.get("llm_used"):
            success_claims += 1
        if claim.metadata.get("llm_fallback_used"):
            fallback_claims += 1
        if claim.metadata.get("llm_cache_hit"):
            cache_hit_claims += 1
        if claim.metadata.get("llm_error"):
            error_counts.update([str(claim.metadata["llm_error"])])
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
        if len(preview_examples) < preview:
            preview_examples.append(
                {
                    "sample_id": sample_id,
                    "text": claim.text,
                    "subject": claim.subject,
                    "relation": claim.relation,
                    "object": claim.object,
                    "qualifier": claim.qualifier,
                    "time": claim.time,
                    "llm_used": claim.metadata.get("llm_used"),
                    "llm_fallback_used": claim.metadata.get("llm_fallback_used"),
                    "llm_cache_hit": claim.metadata.get("llm_cache_hit"),
                    "llm_error": claim.metadata.get("llm_error"),
                    "llm_batch_size": claim.metadata.get("llm_batch_size"),
                }
            )

    elapsed_seconds = round(time.perf_counter() - started_at, 4)
    return {
        "prompt_variant": prompt_variant,
        "configured_batch_size": llm_batch_size,
        "total_claims": total_claims,
        "elapsed_seconds": elapsed_seconds,
        "llm_success_rate": round(safe_ratio(success_claims, total_claims), 4),
        "llm_fallback_rate": round(safe_ratio(fallback_claims, total_claims), 4),
        "llm_cache_hit_rate": round(safe_ratio(cache_hit_claims, total_claims), 4),
        "fill_rates": {
            "structured_fields_present": round(safe_ratio(present_counts["structured"], total_claims), 4),
            "full_triplet_present": round(safe_ratio(present_counts["full_triplet"], total_claims), 4),
            "subject": round(safe_ratio(present_counts["subject"], total_claims), 4),
            "relation": round(safe_ratio(present_counts["relation"], total_claims), 4),
            "object": round(safe_ratio(present_counts["object"], total_claims), 4),
            "qualifier": round(safe_ratio(present_counts["qualifier"], total_claims), 4),
            "time": round(safe_ratio(present_counts["time"], total_claims), 4),
        },
        "top_relations": relation_counts.most_common(10),
        "llm_errors": dict(error_counts),
        "preview_examples": preview_examples,
    }


def main() -> None:
    args = parse_args()
    chunks = load_chunks(args.dataset, args.input, args.offset, args.limit)
    variants = [item.strip() for item in args.prompt_variants.split(",") if item.strip()]
    payload = {
        "dataset": args.dataset,
        "input": args.input,
        "offset": args.offset,
        "evaluated_chunks": len(chunks),
        "variants": [
            summarize_variant(
                chunks=chunks,
                entity_backend=args.entity_backend,
                prompt_variant=variant,
                llm_model=args.llm_model,
                llm_base_url=args.llm_base_url,
                llm_api_style=args.llm_api_style,
                llm_timeout_seconds=args.llm_timeout_seconds,
                llm_batch_size=args.llm_batch_size,
                fallback_to_heuristic=args.fallback_to_heuristic,
                preview=args.preview,
            )
            for variant in variants
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
