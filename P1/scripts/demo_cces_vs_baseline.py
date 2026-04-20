"""Side-by-side demo: CCES λ=0.3 vs top2_span on a real FNC-1 sample.

Picks a sample where the two methods diverge on the predicted label, so
you can SEE why CCES wins. Uses the local DeBERTa-v3-base NLI model.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.data.fnc1 import read_jsonl, sample_to_claim_pair, select_cces_evidence
from p1.nli import build_nli_model
from p1.schemas import Claim, ClaimPair, ClaimSource


def make_pair(sample, headline_evidence: str, label: str) -> ClaimPair:
    sid = sample["sample_id"]
    return ClaimPair(
        claim_a=Claim(
            claim_id=f"{sid}:headline",
            text=sample["headline"].strip(),
            source=ClaimSource(doc_id=f"{sid}:headline", chunk_id="headline"),
            metadata={"variant": label},
        ),
        claim_b=Claim(
            claim_id=f"{sid}:body:{label}",
            text=headline_evidence,
            source=ClaimSource(doc_id=f"body:{sample['body_id']}", chunk_id=sample["body_id"]),
            metadata={"variant": label},
        ),
    )


def short(text: str, n: int = 240) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def main() -> None:
    records = read_jsonl("data/processed/fnc1_train.jsonl")[:600]
    print("Loading DeBERTa-v3-base NLI model (local)...")
    nli = build_nli_model(
        kind="hf",
        model_name="manual_models/DeBERTa-v3-base-mnli-fever-anli",
        bidirectional=True,
    )

    interesting = []
    for s in records:
        top2_pair = sample_to_claim_pair(s, body_mode="top2_span")
        cces_evidence = select_cces_evidence(
            s["headline"], s["body"], k=2, lambda_param=0.3, backend="lexical"
        )
        cces_pair = make_pair(s, cces_evidence, "cces_lambda_0.3")
        top2_res = nli.predict(top2_pair)
        cces_res = nli.predict(cces_pair)
        gold = s["nli_label"]
        # Look for cases where CCES is right and top2_span is wrong
        if cces_res.label.value == gold and top2_res.label.value != gold:
            interesting.append((s, top2_pair, top2_res, cces_pair, cces_res, gold))
        if len(interesting) >= 3:
            break

    print(f"\nFound {len(interesting)} samples where CCES wins and top2_span loses.\n")
    for i, (s, top2_pair, top2_res, cces_pair, cces_res, gold) in enumerate(interesting, 1):
        print("=" * 80)
        print(f"DEMO {i}  (sample_id={s['sample_id']}, gold={gold.upper()})")
        print("=" * 80)
        print(f"HEADLINE: {s['headline']}")
        print(f"\n--- top2_span evidence ---")
        print(f"  text: {short(top2_pair.claim_b.text)}")
        print(
            f"  NLI:  ent={top2_res.entailment_score:.2f}  "
            f"con={top2_res.contradiction_score:.2f}  "
            f"neu={top2_res.neutral_score:.2f}  "
            f"=> PRED: {top2_res.label.value.upper()}  ❌ (gold={gold})"
        )
        print(f"\n--- CCES λ=0.3 evidence ---")
        print(f"  text: {short(cces_pair.claim_b.text)}")
        print(
            f"  NLI:  ent={cces_res.entailment_score:.2f}  "
            f"con={cces_res.contradiction_score:.2f}  "
            f"neu={cces_res.neutral_score:.2f}  "
            f"=> PRED: {cces_res.label.value.upper()}  ✅"
        )
        print()


if __name__ == "__main__":
    main()
