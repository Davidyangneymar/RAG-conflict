"""Export P3 retrieval results in the stable P1 retrieval input contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_retrieval_config
from src.ingestion.fever_loader import load_fever_claims
from src.retrieval.pipeline import RetrievalPipeline
from src.schemas.retrieval import RetrievalQuery
from src.services.evidence_hygiene import apply_evidence_hygiene
from src.services.handoff_adapter import (
    claim_and_response_to_p1_record,
    responses_to_p1_batch,
    retrieval_response_to_p1_record,
)
from src.utils.io import write_json
from src.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export retrieval outputs in the P3 -> P1 handoff contract.")
    parser.add_argument("--config-path", default="config/retrieval.yaml", help="Path to retrieval YAML config.")
    parser.add_argument("--output-path", default=None, help="Optional JSON output path.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of evidence chunks per record.")
    parser.add_argument(
        "--mode",
        choices=["bm25", "dense", "hybrid"],
        default=None,
        help="Retrieval mode. Defaults to the YAML config value.",
    )
    parser.add_argument("--split", default=None, help="Optional dataset split metadata for exported records.")

    parser.add_argument("--query", default=None, help="Single query to export.")
    parser.add_argument("--sample-id", default="manual-query", help="Sample id for single-query export.")
    parser.add_argument("--label", default=None, help="Optional label for single-query export.")

    parser.add_argument("--claims-path", default=None, help="Optional FEVER-style claims JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for claim batch export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_retrieval_config(args.config_path)
    setup_logging(config.log_level)
    pipeline = RetrievalPipeline.from_artifacts(config)
    retrieval_top_k = args.top_k + config.evidence_hygiene_extra_pool if config.enable_evidence_hygiene else args.top_k

    def prepare_response(response):
        cleaned = apply_evidence_hygiene(response, config, top_k=retrieval_top_k)
        if config.enable_diversify:
            diversified = pipeline.diversifier.diversify(
                cleaned.results,
                top_k=args.top_k,
                min_unique_sources=2,
                max_per_source=config.max_per_source,
                prefer_recent=False,
            )
            return cleaned.model_copy(update={"results": diversified})
        return cleaned.model_copy(update={"results": cleaned.results[: args.top_k]})

    if args.query:
        response = pipeline.retrieve(
            RetrievalQuery(
                query=args.query,
                top_k=retrieval_top_k,
                mode=args.mode or config.default_mode,
                use_diversify=False,
            )
        )
        response = prepare_response(response)
        record = retrieval_response_to_p1_record(
            response,
            sample_id=args.sample_id,
            label=args.label,
            metadata={"dataset": "manual"} if args.split is None else {"dataset": "manual", "split": args.split},
        )
        payload = record.model_dump()
    elif args.claims_path:
        claims = load_fever_claims(args.claims_path)
        if args.limit is not None:
            claims = claims[: args.limit]

        records = []
        for claim in claims:
            response = pipeline.retrieve(
                RetrievalQuery(
                    query=claim.query,
                    top_k=retrieval_top_k,
                    mode=args.mode or config.default_mode,
                    use_diversify=False,
                )
            )
            response = prepare_response(response)
            records.append(claim_and_response_to_p1_record(claim, response, split=args.split))

        payload = responses_to_p1_batch(records).model_dump()
    else:
        raise SystemExit("Provide either --query or --claims-path.")

    if args.output_path:
        write_json(args.output_path, payload)
        print(f"Wrote P1 handoff payload to {args.output_path}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
