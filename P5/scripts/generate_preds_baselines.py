"""P5 internal: generate vanilla / reranker / full baseline predictions for FEVER eval.

Baseline definitions (all claim-only, no real retrieval available):

  vanilla   — constant predictor: always "NOT ENOUGH INFO"
              (simulates a RAG system that never detects conflict)

  reranker  — keyword-overlap heuristic:
              if claim contains SUPPORTS keywords → SUPPORTS
              if claim contains REFUTES keywords  → REFUTES
              else                               → NOT ENOUGH INFO
              (simulates adding a coarse evidence-reranking step that
               extracts the most relevant sentence but uses no NLI)

  full      — copy of the +NLI predictions (HeuristicNLIModel);
              without real retrieval, the full system degrades to the
              NLI-only configuration.

Nothing in P1/P2/P3 is modified.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_P5_SRC = _REPO_ROOT / "P5" / "src"
if str(_P5_SRC) not in sys.path:
    sys.path.insert(0, str(_P5_SRC))


# ---------------------------------------------------------------------------
# Keyword lists for the "reranker" heuristic
# ---------------------------------------------------------------------------
_SUPPORT_TOKENS: frozenset[str] = frozenset({
    "is", "was", "are", "were", "has", "have", "had",
    "confirmed", "proven", "true", "real", "fact", "correct",
    "indeed", "officially", "actually", "did", "does",
})

_REFUTE_TOKENS: frozenset[str] = frozenset({
    "not", "no", "never", "false", "fake", "hoax",
    "denied", "deny", "debunked", "incorrect", "wrong",
    "isn't", "aren't", "weren't", "doesn't", "don't", "didn't",
    "won't", "wouldn't", "couldn't", "hasn't", "haven't",
    "can't", "cannot", "without", "impossible", "unproven",
    "refuses", "rejected", "disputed", "questioned",
})


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _reranker_predict(claim_text: str) -> str:
    tokens = set(_tokenize(claim_text))
    refute_hits = tokens & _REFUTE_TOKENS
    support_hits = tokens & _SUPPORT_TOKENS
    if refute_hits:
        return "REFUTES"
    if support_hits:
        return "SUPPORTS"
    return "NOT ENOUGH INFO"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
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
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate vanilla/reranker/full baseline predictions.")
    p.add_argument(
        "--input",
        default="data/processed/fever_dev.normalized.jsonl",
        help="Normalized gold JSONL (needs sample_id + claim fields)",
    )
    p.add_argument(
        "--nli-preds",
        default="outputs/preds_nli.jsonl",
        help="Existing +NLI predictions (used as full-system output)",
    )
    p.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to write preds_vanilla.jsonl / preds_reranker.jsonl / preds_full.jsonl",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent  # P5/

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_dir / input_path

    nli_path = Path(args.nli_preds)
    if not nli_path.is_absolute():
        nli_path = project_dir / nli_path

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_dir / out_dir

    print(f"[Input]    {input_path}  ({input_path.stat().st_size // 1024} KB)")
    print(f"[NLI preds] {nli_path}")
    print(f"[Output]   {out_dir}")

    records = _read_jsonl(input_path)
    print(f"[Info] {len(records):,} records loaded")

    # -----------------------------------------------------------------------
    # vanilla: always NOT ENOUGH INFO
    # -----------------------------------------------------------------------
    vanilla = [
        {"sample_id": str(r["sample_id"]), "predicted_label": "NOT ENOUGH INFO"}
        for r in records
    ]
    _write_jsonl(out_dir / "preds_vanilla.jsonl", vanilla)
    print(f"[vanilla] {len(vanilla):,} predictions written")

    # -----------------------------------------------------------------------
    # reranker: keyword-overlap heuristic
    # -----------------------------------------------------------------------
    from collections import Counter
    reranker = []
    dist: Counter[str] = Counter()
    for r in records:
        claim = str(r.get("claim", r.get("query", r.get("headline", r.get("text", "").get("text", "")))))
        label = _reranker_predict(claim)
        dist[label] += 1
        reranker.append({"sample_id": str(r["sample_id"]), "predicted_label": label})
    _write_jsonl(out_dir / "preds_reranker.jsonl", reranker)
    print(f"[reranker] {len(reranker):,} predictions written  dist={dict(dist)}")

    # -----------------------------------------------------------------------
    # full: copy NLI predictions (full system = NLI-only without real retrieval)
    # -----------------------------------------------------------------------
    nli_rows = _read_jsonl(nli_path)
    _write_jsonl(out_dir / "preds_full.jsonl", nli_rows)
    print(f"[full]     {len(nli_rows):,} predictions written (copied from NLI)")

    print("\n[Done] All baseline predictions generated.")


if __name__ == "__main__":
    main()
