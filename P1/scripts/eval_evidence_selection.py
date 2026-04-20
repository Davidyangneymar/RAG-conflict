"""Evidence-selection ablation: full / best_sentence / top2_span / top3_span vs CCES.

This is the headline experiment for the Claim-Conditioned Evidence Selector
contribution. We hold the NLI model fixed and vary only how the body-side
evidence is constructed, so any Macro-F1 delta is attributable to the
selection strategy, not the classifier.

Output:
    - Per-mode metrics (accuracy, macro_f1, label-wise F1, length stats)
    - Per-mode confusion matrix
    - Optional JSON report for downstream plotting

Run:
    python scripts/eval_evidence_selection.py \
        --models heuristic --limit 600 \
        --modes full best_sentence top2_span cces cces3
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

from p1.blocking import BlockingConfig, MultiStageBlocker
from p1.data.fnc1 import read_jsonl, sample_to_claim_pair, select_cces_evidence
from p1.nli import build_nli_model
from p1.nli_ensemble import EnsembleNLIModel
from p1.schemas import Claim, ClaimPair, ClaimSource, NliLabel


DEFAULT_MODES = ("full", "best_sentence", "top2_span", "top3_span", "cces", "cces3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence-selection ablation on FNC-1.")
    parser.add_argument("--input", default="data/processed/fnc1_train.jsonl")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_MODES),
        help="Body modes to compare. cces/cces3/cces_embed exercise the new selector.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["heuristic"],
        help="NLI backends. With more than one, an equal-weighted ensemble is used.",
    )
    parser.add_argument("--hf-model-name", default="cross-encoder/nli-deberta-v3-large")
    parser.add_argument("--single-direction", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--cces-lambda-sweep",
        nargs="+",
        type=float,
        default=None,
        help="If set, also sweep CCES across these λ values (k=2, lexical backend).",
    )
    parser.add_argument(
        "--cces-k",
        type=int,
        default=2,
        help="k for the cces_sweep mode (number of evidence sentences to select)",
    )
    return parser.parse_args()


def lexical_similarity(pair: ClaimPair) -> float:
    blocker = MultiStageBlocker(config=BlockingConfig(min_lexical_similarity=0.0))
    built = blocker._build_pair(pair.claim_a, pair.claim_b)
    return 0.0 if built is None else built.lexical_similarity


def macro_f1_with_breakdown(confusion: dict[str, Counter[str]]) -> tuple[float, dict[str, float]]:
    labels = [label.value for label in NliLabel]
    per_label_f1: dict[str, float] = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_label_f1[label] = (
            (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        )
    return sum(per_label_f1.values()) / len(per_label_f1), per_label_f1


def _build_pair_with_custom_body(sample: dict, body_text: str, mode_label: str) -> ClaimPair:
    sample_id = sample["sample_id"]
    headline = sample["headline"].strip()
    claim_a = Claim(
        claim_id=f"{sample_id}:headline",
        text=headline,
        source=ClaimSource(doc_id=f"{sample_id}:headline", chunk_id="headline"),
        metadata={"dataset": "fnc1", "body_mode": mode_label},
    )
    claim_b = Claim(
        claim_id=f"{sample_id}:body",
        text=body_text,
        source=ClaimSource(doc_id=f"body:{sample['body_id']}", chunk_id=sample["body_id"]),
        metadata={"dataset": "fnc1", "body_mode": mode_label},
    )
    return ClaimPair(claim_a=claim_a, claim_b=claim_b)


def evaluate_mode(records, body_mode: str, model) -> dict:
    started = time.perf_counter()
    pairs: list[ClaimPair] = []
    body_lengths: list[int] = []

    cces_sweep_match = None
    if body_mode.startswith("cces_lambda_"):
        try:
            cces_sweep_match = float(body_mode.split("cces_lambda_", 1)[1])
        except ValueError:
            cces_sweep_match = None

    for sample in records:
        if cces_sweep_match is not None:
            evidence = select_cces_evidence(
                sample["headline"], sample["body"], k=2,
                lambda_param=cces_sweep_match, backend="lexical",
            )
            pair = _build_pair_with_custom_body(sample, evidence, body_mode)
        else:
            pair = sample_to_claim_pair(sample, body_mode=body_mode)
        pair.lexical_similarity = lexical_similarity(pair)
        pairs.append(pair)
        body_lengths.append(len(pair.claim_b.text.split()))

    results = model.predict_many(pairs)
    confusion: dict[str, Counter[str]] = {label.value: Counter() for label in NliLabel}
    correct = 0
    sample_predictions: list[dict] = []
    for sample, result in zip(records, results):
        gold = sample["nli_label"]
        pred = result.label.value
        confusion[gold][pred] += 1
        if gold == pred:
            correct += 1
        sample_predictions.append(
            {
                "sample_id": sample["sample_id"],
                "gold": gold,
                "pred": pred,
                "entailment_score": round(result.entailment_score, 4),
                "contradiction_score": round(result.contradiction_score, 4),
                "neutral_score": round(result.neutral_score, 4),
            }
        )

    macro_f1, per_label = macro_f1_with_breakdown(confusion)
    return {
        "body_mode": body_mode,
        "n": len(records),
        "accuracy": correct / len(records) if records else 0.0,
        "macro_f1": macro_f1,
        "per_label_f1": per_label,
        "body_length_words_mean": statistics.mean(body_lengths) if body_lengths else 0.0,
        "body_length_words_p90": (
            statistics.quantiles(body_lengths, n=10)[-1] if len(body_lengths) >= 10 else max(body_lengths, default=0)
        ),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "confusion": {gold: dict(row) for gold, row in confusion.items()},
        "predictions": sample_predictions,
    }


def build_model(args: argparse.Namespace):
    backends = [
        build_nli_model(
            kind=kind,
            model_name=args.hf_model_name,
            bidirectional=not args.single_direction,
        )
        for kind in args.models
    ]
    if len(backends) == 1:
        return backends[0]
    return EnsembleNLIModel(models=[(m, 1.0) for m in backends], name="+".join(args.models))


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)[: args.limit]
    print(f"Evaluating {len(records)} samples on {len(args.modes)} body modes")
    print(f"NLI backend: {args.models} (bidirectional={not args.single_direction})")
    model = build_model(args)

    modes = list(args.modes)
    if args.cces_lambda_sweep:
        for lam in args.cces_lambda_sweep:
            modes.append(f"cces_lambda_{lam}")

    rows: list[dict] = []
    for mode in modes:
        print(f"\n--- mode={mode} ---")
        result = evaluate_mode(records, mode, model)
        rows.append(result)
        print(
            f"  acc={result['accuracy']:.4f}  macro_f1={result['macro_f1']:.4f}  "
            f"avg_len={result['body_length_words_mean']:.1f}w  "
            f"runtime={result['duration_seconds']}s"
        )
        per_label = ", ".join(f"{k}={v:.3f}" for k, v in result["per_label_f1"].items())
        print(f"  per-label F1: {per_label}")

    print("\n=== Summary (sorted by macro_f1) ===")
    for row in sorted(rows, key=lambda r: r["macro_f1"], reverse=True):
        delta_vs_top2 = row["macro_f1"] - next(
            (r["macro_f1"] for r in rows if r["body_mode"] == "top2_span"), row["macro_f1"]
        )
        marker = "*" if row["body_mode"].startswith("cces") else " "
        print(
            f"  {marker} {row['body_mode']:14s}  macro_f1={row['macro_f1']:.4f}  "
            f"vs top2_span: {delta_vs_top2:+.4f}  "
            f"avg_len={row['body_length_words_mean']:6.1f}w"
        )

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(
                {
                    "input": args.input,
                    "models": args.models,
                    "n": len(records),
                    "results": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote JSON report to {args.output_json}")


if __name__ == "__main__":
    main()
