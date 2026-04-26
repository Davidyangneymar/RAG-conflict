"""P5 internal: generate +NLI baseline predictions using P1's HeuristicNLIModel.

Strategy: pair each claim against a "positive" version (negation/refutation tokens
stripped) to detect explicit stance signals in the claim text itself.

Label mapping:
  entailment  → SUPPORTS
  contradiction → REFUTES
  neutral     → NOT ENOUGH INFO

Limitation: only claims with EXPLICIT negation/refutation tokens (no, not, never,
false, fake, hoax, denied, debunked, …) trigger REFUTES.  Implicit factual errors
without those tokens will be predicted SUPPORTS.  This is expected for a
retrieval-free NLI baseline.

Nothing in P1/P2/P3 is modified; P1 is imported read-only as a library.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap P1 on the path (read-only import – no P1 files are changed)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_P1_SRC = _REPO_ROOT / "P1" / "src"
if str(_P1_SRC) not in sys.path:
    sys.path.insert(0, str(_P1_SRC))

from p1.nli import HeuristicNLIModel  # noqa: E402
from p1.schemas import Claim, ClaimPair, ClaimSource  # noqa: E402

# ---------------------------------------------------------------------------
# Negation / refutation tokens used to build the "positive" version
# ---------------------------------------------------------------------------
_NEG_TOKENS: frozenset[str] = frozenset(
    {
        "not", "no", "never", "false", "fake", "hoax",
        "denied", "deny", "without", "debunked", "incorrect",
        "incapable", "unable", "unproven", "impossible", "wasn't",
        "isn't", "aren't", "weren't", "doesn't", "don't", "didn't",
        "won't", "wouldn't", "couldn't", "shouldn't", "hasn't", "haven't",
        "hadn't", "can't", "cannot",
    }
)

# Map P1 NliLabel values → FEVER labels
_NLI_TO_FEVER: dict[str, str] = {
    "entailment": "SUPPORTS",
    "contradiction": "REFUTES",
    "neutral": "NOT ENOUGH INFO",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jaccard(a_tokens: list[str], b_tokens: list[str]) -> float:
    a, b = set(a_tokens), set(b_tokens)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _build_pair(claim_text: str, sample_id: str) -> ClaimPair:
    """Build a ClaimPair: original claim vs. a negation-stripped version."""
    a_tokens = claim_text.lower().split()
    # Strip punctuation before checking against NEG_TOKENS
    b_tokens = [t for t in a_tokens if t.rstrip(".,!?;:\"'") not in _NEG_TOKENS]
    positive_text = " ".join(b_tokens) if b_tokens else claim_text

    overlap = _jaccard(a_tokens, b_tokens)

    source = ClaimSource(doc_id=sample_id)
    claim_a = Claim(claim_id=f"{sample_id}-a", text=claim_text, source=source)
    claim_b = Claim(claim_id=f"{sample_id}-b", text=positive_text, source=source)
    return ClaimPair(claim_a=claim_a, claim_b=claim_b, lexical_similarity=overlap)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate +NLI baseline predictions using P1 HeuristicNLIModel."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Normalized JSONL with 'sample_id' and 'query' (or 'claim') fields.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Destination JSONL for predictions (sample_id + predicted_label).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of pairs per NLI batch (default: 512).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    records = _read_jsonl(input_path)
    print(f"[generate_preds_nli] loaded {len(records):,} records from {input_path}")

    model = HeuristicNLIModel(bidirectional=True)

    predictions: list[dict] = []
    batch_size = args.batch_size
    total = len(records)

    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]

        pairs: list[ClaimPair] = []
        for offset, rec in enumerate(batch):
            sample_id = str(rec.get("sample_id", start + offset))
            # Accept both 'query' (normalized format) and 'claim' (raw fever format)
            text = rec.get("query") or rec.get("claim") or ""
            pairs.append(_build_pair(text, sample_id))

        nli_results = model.predict_many(pairs)

        for rec, result in zip(batch, nli_results):
            sample_id = str(rec.get("sample_id", ""))
            fever_label = _NLI_TO_FEVER.get(result.label.value, "NOT ENOUGH INFO")
            predictions.append({"predicted_label": fever_label, "sample_id": sample_id})

        done = min(start + batch_size, total)
        if done % 2000 == 0 or done == total:
            print(f"  {done:,}/{total:,} processed …")

    _write_jsonl(output_path, predictions)
    print(f"[generate_preds_nli] wrote {len(predictions):,} predictions → {output_path}")

    dist = Counter(p["predicted_label"] for p in predictions)
    print("Distribution:", dict(dist))


if __name__ == "__main__":
    main()
