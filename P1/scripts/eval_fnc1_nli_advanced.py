"""Advanced FNC-1 NLI evaluation: ensemble + temperature calibration + thresholds.

Pipeline:
    1. Load FNC-1 normalized JSONL (same as eval_fnc1_nli.py)
    2. Build N base models (heuristic / hf / llm) and assemble an ensemble
    3. Split samples into dev / test
    4. Fit temperature on dev (NLL minimization)
    5. Grid-search class thresholds on dev (macro-F1 maximization)
    6. Evaluate on test, report calibrated and uncalibrated metrics

Designed to run end-to-end with `--models heuristic` (no network) so it
acts as both a research script and a smoke test.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.blocking import BlockingConfig, MultiStageBlocker
from p1.data.fnc1 import read_jsonl, sample_to_claim_pair, select_cces_evidence
from p1.schemas import Claim, ClaimSource
from p1.nli import build_nli_model
from p1.nli_ensemble import (
    CalibratedNLIModel,
    EnsembleNLIModel,
    ThresholdedNLIModel,
    fit_temperature,
    grid_search_thresholds,
)
from p1.schemas import ClaimPair, NliLabel


def _build_pair_for_body_mode(sample: dict, body_mode: str) -> ClaimPair:
    """Like sample_to_claim_pair but supports dynamic cces_lambda_X modes."""
    if body_mode.startswith("cces_lambda_"):
        try:
            lam = float(body_mode.split("cces_lambda_", 1)[1])
        except ValueError:
            lam = 0.3
        evidence = select_cces_evidence(
            sample["headline"], sample["body"], k=2,
            lambda_param=lam, backend="lexical",
        )
        sample_id = sample["sample_id"]
        claim_a = Claim(
            claim_id=f"{sample_id}:headline",
            text=sample["headline"].strip(),
            source=ClaimSource(doc_id=f"{sample_id}:headline", chunk_id="headline"),
            metadata={"dataset": "fnc1", "body_mode": body_mode},
        )
        claim_b = Claim(
            claim_id=f"{sample_id}:body",
            text=evidence,
            source=ClaimSource(doc_id=f"body:{sample['body_id']}", chunk_id=sample["body_id"]),
            metadata={"dataset": "fnc1", "body_mode": body_mode},
        )
        return ClaimPair(claim_a=claim_a, claim_b=claim_b)
    return sample_to_claim_pair(sample, body_mode=body_mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble + calibration eval on FNC-1")
    parser.add_argument("--input", default="data/processed/fnc1_train.jsonl")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--dev-fraction", type=float, default=0.4)
    parser.add_argument(
        "--body-mode",
        default="top2_span",
        help="Body mode; supports any value passed to sample_to_claim_pair "
        "plus dynamic 'cces_lambda_X' (e.g. cces_lambda_0.3).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["heuristic"],
        help="Sequence of NLI kinds to ensemble (e.g. heuristic hf llm)",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional explicit weights matching --models (defaults to uniform)",
    )
    parser.add_argument("--hf-model-name", default="cross-encoder/nli-deberta-v3-large")
    parser.add_argument("--single-direction", action="store_true")
    parser.add_argument(
        "--output-json",
        default=None,
        help="If set, write a structured JSON report to this path",
    )
    return parser.parse_args()


def lexical_similarity(pair: ClaimPair) -> float:
    blocker = MultiStageBlocker(config=BlockingConfig(min_lexical_similarity=0.0))
    built = blocker._build_pair(pair.claim_a, pair.claim_b)
    return 0.0 if built is None else built.lexical_similarity


def macro_f1_from_confusion(confusion: dict[str, Counter[str]]) -> float:
    labels = [label.value for label in NliLabel]
    f1_scores = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return sum(f1_scores) / len(f1_scores)


def evaluate(results, samples) -> dict:
    confusion: dict[str, Counter[str]] = {label.value: Counter() for label in NliLabel}
    correct = 0
    for sample, result in zip(samples, results):
        gold = sample["nli_label"]
        pred = result.label.value
        confusion[gold][pred] += 1
        if gold == pred:
            correct += 1
    total = len(samples)
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1_from_confusion(confusion),
        "confusion": {gold: dict(row) for gold, row in confusion.items()},
        "n": total,
    }


def main() -> None:
    args = parse_args()

    records = read_jsonl(args.input)[: args.limit]
    if args.weights and len(args.weights) != len(args.models):
        raise SystemExit("--weights must have same length as --models")
    weights = args.weights or [1.0] * len(args.models)

    print(f"Loaded {len(records)} samples from {args.input}")
    print(f"Models: {args.models}  Weights: {weights}")

    backends = []
    for kind in args.models:
        backends.append(
            build_nli_model(
                kind=kind,
                model_name=args.hf_model_name,
                bidirectional=not args.single_direction,
            )
        )
    ensemble = EnsembleNLIModel(
        models=list(zip(backends, weights)),
        name="+".join(args.models),
    )

    prepared = []
    for sample in records:
        pair = _build_pair_for_body_mode(sample, args.body_mode)
        pair.lexical_similarity = lexical_similarity(pair)
        prepared.append((sample, pair))

    cutoff = max(1, int(len(prepared) * args.dev_fraction))
    dev_samples = [s for s, _ in prepared[:cutoff]]
    dev_pairs = [p for _, p in prepared[:cutoff]]
    test_samples = [s for s, _ in prepared[cutoff:]]
    test_pairs = [p for _, p in prepared[cutoff:]]
    print(f"Split: dev={len(dev_samples)}  test={len(test_samples)}")

    dev_results = ensemble.predict_many(dev_pairs)
    test_results_raw = ensemble.predict_many(test_pairs)

    raw_metrics = evaluate(test_results_raw, test_samples)
    print("\n=== Raw ensemble (no calibration) ===")
    print(f"  accuracy={raw_metrics['accuracy']:.4f}  macro_f1={raw_metrics['macro_f1']:.4f}")

    dev_triples = [
        (r.entailment_score, r.contradiction_score, r.neutral_score) for r in dev_results
    ]
    dev_gold = [NliLabel(s["nli_label"]) for s in dev_samples]

    temperature = fit_temperature(dev_triples, dev_gold)
    print(f"\nFitted temperature on dev: T={temperature:.3f}")

    calibrated = CalibratedNLIModel(inner=ensemble, temperature=temperature)
    calibrated_dev = calibrated.predict_many(dev_pairs)
    calibrated_dev_triples = [
        (r.entailment_score, r.contradiction_score, r.neutral_score) for r in calibrated_dev
    ]
    ent_threshold, con_threshold, dev_macro_f1 = grid_search_thresholds(
        calibrated_dev_triples, dev_gold
    )
    print(
        f"Best thresholds on dev: entailment>={ent_threshold:.2f}  contradiction>={con_threshold:.2f}  "
        f"(dev macro_f1={dev_macro_f1:.4f})"
    )

    final_model = ThresholdedNLIModel(
        inner=calibrated,
        entailment_threshold=ent_threshold,
        contradiction_threshold=con_threshold,
    )
    test_results_final = final_model.predict_many(test_pairs)
    final_metrics = evaluate(test_results_final, test_samples)
    print("\n=== Final (ensemble + calibration + thresholds) ===")
    print(f"  accuracy={final_metrics['accuracy']:.4f}  macro_f1={final_metrics['macro_f1']:.4f}")

    delta_macro = final_metrics["macro_f1"] - raw_metrics["macro_f1"]
    delta_acc = final_metrics["accuracy"] - raw_metrics["accuracy"]
    print(f"\nLift over raw ensemble: macro_f1 {delta_macro:+.4f}  accuracy {delta_acc:+.4f}")

    print("\nFinal confusion (gold -> pred):")
    for gold_label in sorted(final_metrics["confusion"]):
        row = ", ".join(
            f"{pred}:{count}" for pred, count in sorted(final_metrics["confusion"][gold_label].items())
        )
        print(f"  {gold_label} -> {row}")

    if args.output_json:
        report = {
            "input": args.input,
            "models": args.models,
            "weights": weights,
            "body_mode": args.body_mode,
            "bidirectional": not args.single_direction,
            "dev_n": len(dev_samples),
            "test_n": len(test_samples),
            "temperature": temperature,
            "thresholds": {"entailment": ent_threshold, "contradiction": con_threshold},
            "raw": raw_metrics,
            "final": final_metrics,
            "lift": {"macro_f1": delta_macro, "accuracy": delta_acc},
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {args.output_json}")


if __name__ == "__main__":
    main()
