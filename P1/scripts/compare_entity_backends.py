from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.claim_extraction import extract_entity_candidates
from p1.data.fnc1 import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview regex vs spaCy entity extraction on FNC-1 headlines.")
    parser.add_argument("--input", default="data/processed/fnc1_train.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)[: args.limit]

    for sample in records:
        headline = sample["headline"]
        regex_entities = extract_entity_candidates(headline, backend="regex")
        spacy_entities = extract_entity_candidates(headline, backend="spacy")
        print(f"\n- sample_id: {sample['sample_id']}")
        print(f"  headline: {headline}")
        print(f"  regex: {regex_entities}")
        print(f"  spacy: {spacy_entities}")


if __name__ == "__main__":
    main()
