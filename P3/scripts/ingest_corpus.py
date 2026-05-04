"""CLI entrypoint for corpus ingestion and index building."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_retrieval_config
from src.ingestion.build_index import IndexBuilder
from src.ingestion.loaders import load_documents
from src.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a corpus into BM25 and Qdrant indexes.")
    parser.add_argument("--input-path", required=True, help="Path to the JSONL corpus.")
    parser.add_argument(
        "--loader",
        choices=["generic", "fever_wiki", "averitec_dev", "averitec_dev_smoke"],
        default="generic",
        help="Loader type to use for the input corpus.",
    )
    parser.add_argument("--dataset", default="fever_wiki_sample", help="Dataset label stored in metadata.")
    parser.add_argument(
        "--config-path",
        default="config/retrieval.yaml",
        help="Path to retrieval YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_retrieval_config(args.config_path)
    setup_logging(config.log_level)

    documents = load_documents(args.input_path, loader_type=args.loader, dataset=args.dataset)
    builder = IndexBuilder(config)
    chunks = builder.build(documents)

    print(f"Ingested {len(documents)} documents into {len(chunks)} chunks.")
    print(f"Chunk store: {config.resolve_path(config.chunk_store_path)}")
    print(f"Qdrant path: {config.resolve_path(config.qdrant_path)}")


if __name__ == "__main__":
    main()
