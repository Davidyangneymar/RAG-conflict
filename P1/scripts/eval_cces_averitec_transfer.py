"""CCES cross-dataset transfer: FNC-1 → AVeriTeC.

Validates that the CCES advantage discovered on FNC-1 also holds on a
different fact-checking dataset (AVeriTeC dev split). Treats AVeriTeC's
`claim` as the headline and the concatenation of `questions[*].answers[*].answer`
as the body.

Run:
    python scripts/eval_cces_averitec_transfer.py --limit 200 \
        --modes top2_span cces_lambda_0.3 --output-json data/processed/cces_averitec_transfer.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.data.averitec import load_averitec_records
from p1.data.fnc1 import select_cces_evidence
from p1.evidence_selection import build_evidence_selector  # noqa: F401 (cache warmup)
from p1.nli import build_nli_model
from p1.schemas import Claim, ClaimPair, ClaimSource, NliLabel


def collect_body(record: dict) -> str:
    """Concatenate AVeriTeC question answers into a single body text."""
    parts: list[str] = []
    for q in record.get("questions", []):
        for a in q.get("answers", []):
            ans = (a.get("answer") or "").strip()
            if ans:
                parts.append(ans)
    return " ".join(parts)


def best_sentence_evidence(claim_text: str, body: str) -> str:
    """Mirror FNC-1's best_sentence: pick top-1 lexical-overlap sentence."""
    import re
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    if not sents:
        return body
    claim_words = set(re.findall(r"\w+", claim_text.lower()))
    if not claim_words:
        return sents[0]
    scored = []
    for s in sents:
        sw = set(re.findall(r"\w+", s.lower()))
        if not sw:
            scored.append((0.0, s))
            continue
        overlap = len(sw & claim_words) / len(sw | claim_words)
        scored.append((overlap, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def top2_span_evidence(claim_text: str, body: str) -> str:
    """Pick best 2 contiguous sentences by lexical overlap with claim."""
    import re
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    if len(sents) <= 2:
        return body
    claim_words = set(re.findall(r"\w+", claim_text.lower()))
    if not claim_words:
        return " ".join(sents[:2])
    best_idx, best_score = 0, -1.0
    for i in range(len(sents) - 1):
        span_words = set(re.findall(r"\w+", (sents[i] + " " + sents[i+1]).lower()))
        score = (len(span_words & claim_words) / len(span_words | claim_words)) if span_words else 0.0
        if score > best_score:
            best_score, best_idx = score, i
    return sents[best_idx] + " " + sents[best_idx + 1]


def make_pair(record: dict, evidence: str, mode: str) -> ClaimPair:
    sid = record["sample_id"]
    return ClaimPair(
        claim_a=Claim(
            claim_id=f"{sid}:claim",
            text=record["claim"].strip(),
            source=ClaimSource(doc_id=sid, chunk_id="claim"),
            metadata={"dataset": "averitec", "mode": mode},
        ),
        claim_b=Claim(
            claim_id=f"{sid}:body:{mode}",
            text=evidence,
            source=ClaimSource(doc_id=f"{sid}:body", chunk_id="body"),
            metadata={"dataset": "averitec", "mode": mode},
        ),
    )


def evidence_for_mode(record: dict, body: str, mode: str) -> str:
    if mode == "top2_span":
        return top2_span_evidence(record["claim"], body)
    if mode == "best_sentence":
        return best_sentence_evidence(record["claim"], body)
    if mode == "full":
        return body
    if mode.startswith("cces_lambda_"):
        try:
            lam = float(mode.split("cces_lambda_", 1)[1])
        except ValueError:
            lam = 0.3
        return select_cces_evidence(record["claim"], body, k=2, lambda_param=lam, backend="lexical")
    if mode == "cces":
        return select_cces_evidence(record["claim"], body, k=2, lambda_param=0.7, backend="lexical")
    raise ValueError(f"Unknown mode: {mode}")


def macro_f1_from_confusion(confusion: dict[str, Counter[str]]) -> tuple[float, dict[str, float]]:
    labels = [l.value for l in NliLabel]
    per: dict[str, float] = {}
    for l in labels:
        tp = confusion[l][l]
        fp = sum(confusion[o][l] for o in labels if o != l)
        fn = sum(confusion[l][o] for o in labels if o != l)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        per[l] = (2 * p * r / (p + r)) if (p + r) else 0.0
    return sum(per.values()) / len(per), per


def evaluate_mode(records, bodies, mode, model) -> dict:
    started = time.perf_counter()
    pairs = [make_pair(r, evidence_for_mode(r, b, mode), mode) for r, b in zip(records, bodies)]
    results = model.predict_many(pairs)
    confusion = {l.value: Counter() for l in NliLabel}
    correct = 0
    preds = []
    for r, res in zip(records, results):
        gold = r["nli_label"]
        pred = res.label.value
        confusion[gold][pred] += 1
        if gold == pred:
            correct += 1
        preds.append({"sample_id": r["sample_id"], "gold": gold, "pred": pred})
    macro, per_label = macro_f1_from_confusion(confusion)
    body_lens = [len(p.claim_b.text.split()) for p in pairs]
    return {
        "mode": mode,
        "n": len(records),
        "accuracy": correct / len(records),
        "macro_f1": macro,
        "per_label_f1": per_label,
        "body_length_words_mean": statistics.mean(body_lens),
        "duration_seconds": round(time.perf_counter() - started, 2),
        "confusion": {g: dict(row) for g, row in confusion.items()},
        "predictions": preds,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/averitec/dev.json")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--modes", nargs="+", default=["top2_span", "best_sentence", "cces_lambda_0.3", "cces_lambda_0.5", "cces_lambda_0.7"])
    p.add_argument("--hf-model-name", default="manual_models/DeBERTa-v3-base-mnli-fever-anli")
    p.add_argument("--single-direction", action="store_true")
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_averitec_records(args.input, limit=args.limit)
    records = [r for r in raw if r["claim"] and any(q.get("answers") for q in r.get("questions", []))]
    bodies = [collect_body(r) for r in records]
    nonempty = [(r, b) for r, b in zip(records, bodies) if b.strip()]
    records = [r for r, _ in nonempty]
    bodies = [b for _, b in nonempty]
    print(f"Loaded {len(records)} AVeriTeC records (with non-empty body)")

    label_dist = Counter(r["nli_label"] for r in records)
    print(f"Label distribution: {dict(label_dist)}")

    print(f"Building NLI model: {args.hf_model_name}")
    model = build_nli_model(
        kind="hf", model_name=args.hf_model_name,
        bidirectional=not args.single_direction,
    )

    rows = []
    for mode in args.modes:
        print(f"\n--- mode={mode} ---")
        result = evaluate_mode(records, bodies, mode, model)
        rows.append(result)
        per = ", ".join(f"{k}={v:.3f}" for k, v in result["per_label_f1"].items())
        print(f"  acc={result['accuracy']:.4f}  macro_f1={result['macro_f1']:.4f}  avg_len={result['body_length_words_mean']:.1f}w  {result['duration_seconds']}s")
        print(f"  per-label F1: {per}")

    print("\n=== Summary (sorted by macro_f1) ===")
    base = next((r["macro_f1"] for r in rows if r["mode"] == "top2_span"), rows[0]["macro_f1"])
    for r in sorted(rows, key=lambda x: x["macro_f1"], reverse=True):
        delta = r["macro_f1"] - base
        marker = "*" if r["mode"].startswith("cces") else " "
        print(f"  {marker} {r['mode']:20s}  macro_f1={r['macro_f1']:.4f}  vs top2_span: {delta:+.4f}")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps({"input": args.input, "n": len(records), "results": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
