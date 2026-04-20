from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.pipeline import build_pipeline
from p1.schemas import ChunkInput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local P1 demo pipeline.")
    parser.add_argument("--extractor-kind", choices=["sentence", "structured", "structured_llm"], default="sentence")
    parser.add_argument("--entity-backend", choices=["auto", "regex", "spacy"], default="auto")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--prompt-variant", choices=["baseline", "headline_aware"], default="baseline")
    parser.add_argument("--fallback-to-heuristic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = [
        ChunkInput(
            doc_id="doc_a",
            chunk_id="chunk_1",
            text=(
                "WHO reported that the vaccine reduced severe outcomes in adults. "
                "The study followed patients for six months."
            ),
        ),
        ChunkInput(
            doc_id="doc_b",
            chunk_id="chunk_1",
            text=(
                "A later report said the vaccine did not reduce severe outcomes in adults. "
                "Researchers cited a different population."
            ),
        ),
        ChunkInput(
            doc_id="doc_c",
            chunk_id="chunk_1",
            text="The city opened a new metro line in 2024.",
        ),
    ]

    extractor_options: dict[str, object] = {}
    if args.extractor_kind == "structured_llm":
        extractor_options = {
            "llm_model": args.llm_model,
            "llm_base_url": args.llm_base_url,
            "prompt_variant": args.prompt_variant,
            "fallback_to_heuristic": args.fallback_to_heuristic,
        }

    pipeline = build_pipeline(
        extractor_kind=args.extractor_kind,
        entity_backend=args.entity_backend,
        extractor_options=extractor_options,
    )
    result = pipeline.run(chunks)

    print(f"=== Pipeline Config ===")
    print(
        json.dumps(
            {
                "extractor_kind": args.extractor_kind,
                "entity_backend": args.entity_backend,
                "extractor_options": extractor_options,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=== Claims ===")
    for claim in result.claims:
        print(json.dumps(claim.model_dump(), ensure_ascii=False, indent=2))

    print("\n=== Candidate Pairs ===")
    for pair in result.candidate_pairs:
        print(json.dumps(pair.model_dump(), ensure_ascii=False, indent=2))

    print("\n=== NLI Results ===")
    for item in result.nli_results:
        print(json.dumps(item.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
