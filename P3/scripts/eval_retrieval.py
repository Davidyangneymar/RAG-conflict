"""Lightweight retrieval evaluation over FEVER-style claim samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_retrieval_config
from src.ingestion.fever_loader import load_fever_claims
from src.retrieval.pipeline import RetrievalPipeline
from src.schemas.retrieval import RetrievalQuery
from src.utils.logging import setup_logging
from src.utils.text import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval over FEVER claim samples.")
    parser.add_argument("--claims-path", required=True, help="Path to FEVER-style claims JSONL.")
    parser.add_argument("--config-path", default="config/retrieval.yaml", help="Path to retrieval YAML config.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to inspect per claim.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick evaluation.")
    parser.add_argument(
        "--mode",
        choices=["bm25", "dense", "hybrid"],
        default=None,
        help="Retrieval mode. Defaults to the YAML config value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_retrieval_config(args.config_path)
    setup_logging(config.log_level)
    pipeline = RetrievalPipeline.from_artifacts(config)
    claims = load_fever_claims(args.claims_path)
    if args.limit is not None:
        claims = claims[: args.limit]

    recall_hits: list[float] = []
    unique_source_counts: list[float] = []
    average_score_summaries: list[float] = []

    for claim in claims:
        response = pipeline.retrieve(
            RetrievalQuery(
                query=claim.query,
                top_k=args.top_k,
                mode=args.mode or config.default_mode,
            )
        )

        retrieved_titles = {result.title for result in response.results if result.title}
        recall_hits.append(1.0 if any(title in retrieved_titles for title in claim.evidence_titles) else 0.0)
        unique_source_counts.append(float(len({result.source_name or result.title or result.doc_id for result in response.results})))
        average_score_summaries.append(
            mean(
                [
                    result.score_rerank
                    or result.score_hybrid
                    or result.score_dense
                    or result.score_sparse
                    or 0.0
                    for result in response.results
                ]
            )
        )

    print("Evaluation summary")
    print(f"claims_evaluated: {len(claims)}")
    print(f"retrieval_recall_proxy@{args.top_k}: {mean(recall_hits):.4f}")
    print(f"avg_unique_sources@{args.top_k}: {mean(unique_source_counts):.4f}")
    print(f"avg_topk_score_summary: {mean(average_score_summaries):.4f}")


if __name__ == "__main__":
    main()
