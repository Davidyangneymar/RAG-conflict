from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fnc1_bert_stance_module import (
    DEFAULT_BERT_OUTPUT_DIR,
    BertStancePredictor,
    predict_stance,
    predict_stance_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test for the frozen BERT stance inference module."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_BERT_OUTPUT_DIR,
        help="Directory containing model/runtime artifacts.",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="",
        help="Optional output CSV path for batch predictions.",
    )
    return parser.parse_args()


def get_smoke_samples() -> list[tuple[str, str]]:
    return [
        (
            "Scientists report rising sea levels",
            "A new climate report says global sea levels increased over the last decade. "
            "The report attributes this trend to glacier melt and thermal expansion. "
            "Researchers warn that coastal flooding risk will rise.",
        ),
        (
            "Tech company denies bankruptcy rumors",
            "The company published its quarterly report with positive cash flow. "
            "Executives said there are no plans to file for bankruptcy. "
            "Analysts said the balance sheet remains healthy.",
        ),
        (
            "Local team wins championship",
            "Astronomers discovered a distant exoplanet with unusual atmospheric patterns. "
            "The findings were published in a peer-reviewed journal. "
            "Researchers will continue monitoring the planet.",
        ),
    ]


def validate_single_result(result: dict) -> None:
    required_keys = {
        "claim",
        "body_first3sent",
        "pred_label_4way",
        "pred_label_3way_a",
        "pred_label_3way_a_with_filter",
        "is_filtered_3way_a",
        "decision_score",
        "mapping_scheme",
    }
    missing = sorted(required_keys - set(result.keys()))
    if missing:
        raise AssertionError(f"Missing keys in single inference result: {missing}")


def validate_batch_results(results: list[dict], expected_size: int) -> None:
    if len(results) != expected_size:
        raise AssertionError(f"Expected {expected_size} results, got {len(results)}")
    for row in results:
        validate_single_result(row)


def main() -> None:
    args = parse_args()

    samples = get_smoke_samples()
    claims = [item[0] for item in samples]
    evidence_texts = [item[1] for item in samples]

    predictor = BertStancePredictor.from_output_dir(output_dir=args.output_dir)

    single_result = predictor.predict_stance(claim=claims[0], evidence_text=evidence_texts[0])
    validate_single_result(single_result)

    batch_results = predictor.predict_stance_batch(claims=claims, evidence_texts=evidence_texts)
    validate_batch_results(batch_results, expected_size=len(samples))

    # Verify module-level helper interfaces as well.
    single_from_helper = predict_stance(
        claim=claims[1],
        evidence_text=evidence_texts[1],
        output_dir=args.output_dir,
    )
    validate_single_result(single_from_helper)

    batch_from_helper = predict_stance_batch(
        claims=claims,
        evidence_texts=evidence_texts,
        output_dir=args.output_dir,
    )
    validate_batch_results(batch_from_helper, expected_size=len(samples))

    batch_df = pd.DataFrame(batch_results)

    if args.save_csv.strip():
        output_csv = Path(args.save_csv).resolve()
    else:
        output_csv = (Path(args.output_dir) / "smoke_test_predictions.csv").resolve()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    batch_df.to_csv(output_csv, index=False)

    print("===== BERT Stance Module Smoke Test Passed =====")
    print(f"artifacts: {Path(args.output_dir).resolve()}")
    print("single prediction sample:")
    print(single_result)
    print("\nbatch prediction sample (head):")
    print(batch_df.head(3).to_string(index=False))
    print(f"\nsmoke csv: {output_csv}")


if __name__ == "__main__":
    main()
