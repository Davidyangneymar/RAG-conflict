"""Run a single retrieval query from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_retrieval_config
from src.retrieval.pipeline import RetrievalPipeline
from src.schemas.retrieval import RetrievalQuery
from src.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a retrieval query against the local index.")
    parser.add_argument("--query", required=True, help="Claim or query string to search.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final results to return.")
    parser.add_argument(
        "--mode",
        choices=["bm25", "dense", "hybrid"],
        default=None,
        help="Retrieval mode. Defaults to the YAML config value.",
    )
    parser.add_argument("--config-path", default="config/retrieval.yaml", help="Path to retrieval YAML config.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking.")
    parser.add_argument("--no-diversify", action="store_true", help="Disable diversification.")
    return parser.parse_args()


def format_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    config = load_retrieval_config(args.config_path)
    setup_logging(config.log_level)
    pipeline = RetrievalPipeline.from_artifacts(config)

    request = RetrievalQuery(
        query=args.query,
        top_k=args.top_k,
        mode=args.mode or config.default_mode,
        use_rerank=not args.no_rerank,
        use_diversify=not args.no_diversify,
    )
    response = pipeline.retrieve(request)

    print(f"Query: {response.query}")
    print(f"Mode: {response.mode}")
    print("-" * 80)
    for result in response.results:
        citation = result.title or result.doc_id
        if result.source_url:
            citation = f"{citation} | {result.source_url}"
        print(f"[{result.rank}] {citation}")
        print(
            "scores:"
            f" sparse={format_score(result.score_sparse)}"
            f" dense={format_score(result.score_dense)}"
            f" hybrid={format_score(result.score_hybrid)}"
            f" rerank={format_score(result.score_rerank)}"
        )
        print(result.text[:400].strip())
        print("-" * 80)


if __name__ == "__main__":
    main()
