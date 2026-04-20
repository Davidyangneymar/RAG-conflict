from __future__ import annotations

import argparse
import json
import sys
import time
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
    parser = argparse.ArgumentParser(description="Report runtime/cache behavior for the structured LLM extractor.")
    parser.add_argument("--dataset", choices=["fnc1", "averitec"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--prompt-variant", choices=["baseline", "headline_aware"], default="baseline")
    parser.add_argument("--entity-backend", choices=["auto", "regex", "spacy"], default="spacy")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-style", choices=["chat_completions", "responses"], default="chat_completions")
    parser.add_argument("--llm-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--llm-batch-size", type=int, default=1)
    return parser.parse_args()


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


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> None:
    args = parse_args()
    chunks = load_chunks(args.dataset, args.input, args.offset, args.limit)
    extractor = build_claim_extractor(
        kind="structured_llm",
        entity_backend=args.entity_backend,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_style=args.llm_api_style,
        prompt_variant=args.prompt_variant,
        llm_timeout_seconds=args.llm_timeout_seconds,
        llm_batch_size=args.llm_batch_size,
    )

    started_at = time.perf_counter()
    total_claims = 0
    llm_used = 0
    cache_hits = 0
    failures = 0
    preview: list[dict[str, object]] = []

    observed_batch_sizes: set[int] = set()
    chunk_lookup = {f"{chunk.doc_id}:{chunk.chunk_id or 'chunk0'}": sample_id for sample_id, chunk in chunks}
    for claim in extractor.extract_many([chunk for _, chunk in chunks]):
        sample_id = chunk_lookup.get(f"{claim.source_doc_id}:{claim.source_chunk_id or 'chunk0'}", claim.source_doc_id or "")
        total_claims += 1
        if claim.metadata.get("llm_used"):
            llm_used += 1
        if claim.metadata.get("llm_cache_hit"):
            cache_hits += 1
        if claim.metadata.get("llm_error"):
            failures += 1
        batch_size = claim.metadata.get("llm_batch_size")
        if isinstance(batch_size, int):
            observed_batch_sizes.add(batch_size)
        if len(preview) < 5:
            preview.append(
                {
                    "sample_id": sample_id,
                    "relation": claim.relation,
                    "llm_cache_hit": claim.metadata.get("llm_cache_hit"),
                    "llm_error": claim.metadata.get("llm_error"),
                    "llm_batch_size": claim.metadata.get("llm_batch_size"),
                }
            )

    elapsed_seconds = round(time.perf_counter() - started_at, 4)
    payload = {
        "dataset": args.dataset,
        "input": args.input,
        "offset": args.offset,
        "limit": args.limit,
        "prompt_variant": args.prompt_variant,
        "configured_batch_size": args.llm_batch_size,
        "elapsed_seconds": elapsed_seconds,
        "total_claims": total_claims,
        "llm_success_rate": round(safe_ratio(llm_used, total_claims), 4),
        "cache_hit_rate": round(safe_ratio(cache_hits, total_claims), 4),
        "failure_rate": round(safe_ratio(failures, total_claims), 4),
        "observed_batch_sizes": sorted(observed_batch_sizes),
        "preview": preview,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
