from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.claim_extraction import build_claim_extractor
from p1.schemas import ChunkInput


def main() -> None:
    extractor = build_claim_extractor(kind="structured", entity_backend="spacy")
    chunk = ChunkInput(
        doc_id="demo-doc",
        chunk_id="chunk-0",
        text=(
            "Report: ISIS Leader Abu Bakr al-Baghdadi was reportedly killed in a U.S. airstrike. "
            "Police confirmed the suspect was arrested in New York on October 3, 2014."
        ),
        metadata={"source": "demo"},
    )
    claims = extractor.extract(chunk)
    payload = [
        {
            "claim_id": claim.claim_id,
            "text": claim.text,
            "subject": claim.subject,
            "relation": claim.relation,
            "object": claim.object,
            "qualifier": claim.qualifier,
            "entities": claim.entities,
            "metadata": claim.metadata,
        }
        for claim in claims
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
