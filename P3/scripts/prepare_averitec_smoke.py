"""Prepare a small AVeriTeC dev-only smoke corpus and claim slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.averitec_loader import load_averitec_dev_claims, load_averitec_dev_documents
from src.utils.io import ensure_dir, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AVeriTeC dev smoke corpus/claims JSONL files.")
    parser.add_argument("--input-path", default="data/raw/averitec/fever7/dev.json")
    parser.add_argument("--output-dir", default="data/processed/averitec")
    parser.add_argument("--claim-limit", type=int, default=50)
    parser.add_argument("--max-documents", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    claims = load_averitec_dev_claims(args.input_path, dataset="averitec_dev_smoke")[: args.claim_limit]
    allowed_claim_ids = {claim.claim_id for claim in claims}
    documents = [
        document
        for document in load_averitec_dev_documents(args.input_path, dataset="averitec_dev_smoke")
        if str(document.metadata.get("claim_id")) in allowed_claim_ids
    ][: args.max_documents]

    corpus_path = output_dir / "averitec_dev_smoke_corpus.jsonl"
    claims_path = output_dir / "averitec_dev_smoke_claims.jsonl"
    summary_path = output_dir / "averitec_dev_smoke_summary.json"

    write_jsonl(corpus_path, (document.model_dump() for document in documents))
    write_jsonl(
        claims_path,
        (
            {
                "claim_id": claim.claim_id,
                "claim": claim.query,
                "label": claim.label,
                **claim.metadata,
            }
            for claim in claims
        ),
    )

    summary = {
        "input_path": args.input_path,
        "claim_count": len(claims),
        "document_count": len(documents),
        "corpus_path": str(corpus_path),
        "claims_path": str(claims_path),
        "note": "Dev-only smoke slice built from AVeriTeC QA answers/justifications, not the full evidence collection.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(documents)} smoke documents to {corpus_path}")
    print(f"Wrote {len(claims)} smoke claims to {claims_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
