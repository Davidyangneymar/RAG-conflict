from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.claim_extraction import _load_spacy_model, extract_entity_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether spaCy NER is active and compare extracted entities.")
    parser.add_argument(
        "--text",
        default="Barack Obama visited New York in 2014.",
        help="Text used for the quick backend sanity check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nlp = _load_spacy_model()
    print(f"spacy_model_loaded={nlp is not None}")
    print(f"regex_entities={extract_entity_candidates(args.text, backend='regex')}")
    print(f"auto_entities={extract_entity_candidates(args.text, backend='auto')}")
    print(f"spacy_entities={extract_entity_candidates(args.text, backend='spacy')}")


if __name__ == "__main__":
    main()
