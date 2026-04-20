from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.blocking import BlockingConfig, MultiStageBlocker
from p1.data.fnc1 import rank_body_sentences, read_jsonl, sample_to_claim_pair
from p1.nli import build_nli_model
from p1.schemas import ClaimPair, NliLabel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the current P1 NLI baseline on normalized FNC-1 data.")
    parser.add_argument(
        "--input",
        default="data/processed/fnc1_train.jsonl",
        help="Path to normalized FNC-1 JSONL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of mismatched samples to print",
    )
    parser.add_argument(
        "--body-mode",
        choices=["full", "best_sentence", "top2_span", "top3_span", "cces", "cces3", "cces_embed"],
        default="top2_span",
        help="How to build the body-side claim for FNC-1 evaluation. cces/cces3/cces_embed enable the Claim-Conditioned Evidence Selector (MMR).",
    )
    parser.add_argument(
        "--model",
        choices=["heuristic", "hf"],
        default="heuristic",
        help="Which NLI backend to use for evaluation",
    )
    parser.add_argument(
        "--hf-model-name",
        default="cross-encoder/nli-deberta-v3-large",
        help="HuggingFace model name or local model directory used when --model hf",
    )
    parser.add_argument(
        "--single-direction",
        action="store_true",
        help="Disable bidirectional NLI and only score claim_a -> claim_b",
    )
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_macro_f1(confusion: dict[str, Counter[str]]) -> float:
    labels = [label.value for label in NliLabel]
    f1_scores: list[float] = []

    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)

        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append((2 * precision * recall) / (precision + recall))

    return sum(f1_scores) / len(f1_scores)


def lexical_similarity(pair: ClaimPair) -> float:
    blocker = MultiStageBlocker(config=BlockingConfig(min_lexical_similarity=0.0))
    built = blocker._build_pair(pair.claim_a, pair.claim_b)  # baseline utility reuse
    if built is None:
        return 0.0
    return built.lexical_similarity


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)[: args.limit]
    try:
        model = build_nli_model(
            kind=args.model,
            model_name=args.hf_model_name,
            bidirectional=not args.single_direction,
        )
    except Exception as exc:
        print(f"Failed to initialize model '{args.model}': {exc}")
        print("Tip: use `--model heuristic` for the lightweight local baseline.")
        raise SystemExit(1) from exc

    prepared_pairs = []
    for sample in records:
        pair = sample_to_claim_pair(sample, body_mode=args.body_mode)
        pair.lexical_similarity = lexical_similarity(pair)
        prepared_pairs.append((sample, pair))

    results = model.predict_many([pair for _, pair in prepared_pairs])

    confusion: dict[str, Counter[str]] = {label.value: Counter() for label in NliLabel}
    gold_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    mismatches: list[dict[str, str | float]] = []

    observed_nli_batch_sizes: Counter[int] = Counter()

    for (sample, pair), result in zip(prepared_pairs, results):
        batch_size = result.metadata.get("nli_batch_size")
        if isinstance(batch_size, int):
            observed_nli_batch_sizes.update([batch_size])

        gold = sample["nli_label"]
        pred = result.label.value
        confusion[gold][pred] += 1
        gold_counts.update([gold])
        pred_counts.update([pred])

        if gold != pred and len(mismatches) < args.preview:
            ranked = rank_body_sentences(sample["headline"], sample["body"], top_k=1)
            selected_sentence = ranked[0]["sentence"] if ranked else sample["body"]
            mismatches.append(
                {
                    "sample_id": sample["sample_id"],
                    "gold": gold,
                    "pred": pred,
                    "stance_label": sample["stance_label"],
                    "lexical_similarity": pair.lexical_similarity,
                    "headline": sample["headline"][:180],
                    "body_preview": selected_sentence[:220],
                }
            )

    total = sum(gold_counts.values())
    correct = sum(confusion[label][label] for label in confusion)
    accuracy = safe_ratio(correct, total)
    macro_f1 = compute_macro_f1(confusion)

    print(f"Input: {args.input}")
    print(f"Evaluated samples: {total}")
    print(f"Body mode: {args.body_mode}")
    print(f"Model: {args.model}")
    print(f"Bidirectional: {not args.single_direction}")
    if observed_nli_batch_sizes:
        print(f"Observed NLI batch sizes: {dict(sorted(observed_nli_batch_sizes.items()))}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nGold label distribution:")
    for label, count in sorted(gold_counts.items()):
        print(f"  {label}: {count}")

    print("\nPredicted label distribution:")
    for label, count in sorted(pred_counts.items()):
        print(f"  {label}: {count}")

    print("\nConfusion counts (gold -> pred):")
    for gold_label in sorted(confusion):
        row = ", ".join(f"{pred}:{confusion[gold_label][pred]}" for pred in sorted(confusion[gold_label]))
        print(f"  {gold_label} -> {row}")

    if mismatches:
        print("\nSample mismatches:")
        for mismatch in mismatches:
            print(f"\n- sample_id: {mismatch['sample_id']}")
            print(f"  gold={mismatch['gold']} pred={mismatch['pred']} stance={mismatch['stance_label']}")
            print(f"  lexical_similarity={mismatch['lexical_similarity']}")
            print(f"  headline={mismatch['headline']}")
            print(f"  body_preview={mismatch['body_preview']}")


if __name__ == "__main__":
    main()
