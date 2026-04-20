"""Lightweight fragmentation analysis over P1 benchmark-exported claims."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def is_obviously_broken(claim: dict) -> bool:
    text = (claim.get("text") or "").strip()
    relation = claim.get("relation")
    object_value = claim.get("object")
    tokens = token_count(text)
    starts_bad = bool(text) and (text[0].islower() or not text[0].isalnum())
    ends_bad = bool(text) and text[-1] not in ".!?"
    marker_noise = "-LRB-" in text or "-RRB-" in text or "-LSB-" in text or "-RSB-" in text
    fragment_like = relation is None and object_value is None and tokens < 8
    return fragment_like and (starts_bad or ends_bad or marker_noise or tokens < 4)


def is_entity_only_like(claim: dict) -> bool:
    text = (claim.get("text") or "").strip()
    relation = claim.get("relation")
    object_value = claim.get("object")
    tokens = token_count(text)
    return relation is None and object_value is None and tokens <= 4


def is_title_only_like(claim: dict) -> bool:
    text = (claim.get("text") or "").strip()
    relation = claim.get("relation")
    object_value = claim.get("object")
    subject = (claim.get("subject") or "").strip()
    if relation is not None or object_value is not None or not text or not subject:
        return False
    text_tokens = TOKEN_RE.findall(text.lower())
    subject_tokens = TOKEN_RE.findall(subject.lower())
    if not text_tokens or not subject_tokens:
        return False
    overlap = len(set(text_tokens) & set(subject_tokens))
    return token_count(text) <= 6 and overlap >= max(1, min(len(subject_tokens), len(text_tokens)) - 1)


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze(records: list[dict], max_examples: int) -> dict:
    all_claims: list[tuple[dict, dict]] = []
    short_claims = 0
    broken_claims = 0
    entity_like_claims = 0
    title_like_claims = 0
    claim_token_lengths: list[int] = []
    records_with_broken = 0
    examples: list[dict] = []

    for record in records:
        claims = record.get("claims") or []
        record_has_broken = False
        for claim in claims:
            all_claims.append((record, claim))
            tokens = token_count(claim.get("text") or "")
            claim_token_lengths.append(tokens)
            if tokens < 6:
                short_claims += 1

            broken = is_obviously_broken(claim)
            entity_like = is_entity_only_like(claim)
            title_like = is_title_only_like(claim)

            if broken:
                broken_claims += 1
                record_has_broken = True
            if entity_like:
                entity_like_claims += 1
            if title_like:
                title_like_claims += 1

            if (broken or entity_like or title_like) and len(examples) < max_examples:
                reasons = []
                if broken:
                    reasons.append("broken_fragment")
                if entity_like:
                    reasons.append("entity_only_like")
                if title_like:
                    reasons.append("title_only_like")
                examples.append(
                    {
                        "sample_id": record.get("sample_id"),
                        "claim_id": claim.get("claim_id"),
                        "text": claim.get("text"),
                        "subject": claim.get("subject"),
                        "relation": claim.get("relation"),
                        "object": claim.get("object"),
                        "reasons": reasons,
                        "token_count": tokens,
                    }
                )
        if record_has_broken:
            records_with_broken += 1

    total_claims = len(all_claims)
    total_records = len(records)
    average_claim_token_length = round(sum(claim_token_lengths) / total_claims, 4) if total_claims else 0.0

    return {
        "records_evaluated": total_records,
        "claims_evaluated": total_claims,
        "average_claim_token_length": average_claim_token_length,
        "short_claim_count": short_claims,
        "short_claim_ratio": round(short_claims / total_claims, 4) if total_claims else 0.0,
        "broken_claim_count": broken_claims,
        "broken_claim_ratio": round(broken_claims / total_claims, 4) if total_claims else 0.0,
        "entity_only_like_count": entity_like_claims,
        "entity_only_like_ratio": round(entity_like_claims / total_claims, 4) if total_claims else 0.0,
        "title_only_like_count": title_like_claims,
        "title_only_like_ratio": round(title_like_claims / total_claims, 4) if total_claims else 0.0,
        "records_with_broken_claims": records_with_broken,
        "records_with_broken_claims_ratio": round(records_with_broken / total_records, 4) if total_records else 0.0,
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze claim fragmentation in P1 benchmark exports.")
    parser.add_argument("--input", required=True, help="Path to P1 benchmark export JSONL.")
    parser.add_argument("--output", required=True, help="Path to write analysis JSON.")
    parser.add_argument("--max-examples", type=int, default=10, help="Maximum number of bad examples to retain.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(Path(args.input))
    payload = analyze(records, max_examples=args.max_examples)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
