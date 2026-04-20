from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.claim_extraction import build_claim_extractor
from p1.data.fnc1 import read_jsonl
from p1.schemas import ChunkInput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview structured claim extraction on FNC-1 headlines.")
    parser.add_argument("--input", default="data/processed/fnc1_train.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)[: args.limit]
    extractor = build_claim_extractor(kind="structured", entity_backend="spacy")
    payload: list[dict[str, object]] = []

    for sample in records:
        chunk = ChunkInput(
            doc_id=sample["sample_id"],
            chunk_id="headline",
            text=sample["headline"],
            metadata={"stance_label": sample["stance_label"], "dataset": "fnc1"},
        )
        claims = extractor.extract(chunk)
        payload.append(
            {
                "sample_id": sample["sample_id"],
                "headline": sample["headline"],
                "claims": [
                    {
                        "claim_id": claim.claim_id,
                        "text": claim.text,
                        "subject": claim.subject,
                        "relation": claim.relation,
                        "object": claim.object,
                        "qualifier": claim.qualifier,
                        "time": claim.time,
                        "entities": claim.entities,
                    }
                    for claim in claims
                ],
            }
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
