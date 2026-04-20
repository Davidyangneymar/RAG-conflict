from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fnc1_stance_module import (
    StanceTrainingConfig,
    TEXT_MODE_CHOICES,
    TRAINABLE_BACKEND_CHOICES,
    predict_stance,
    run_training_pipeline,
)


def parse_args() -> StanceTrainingConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Modular FNC-1 stance classification training script. "
            "Default best baseline: TF-IDF LinearSVM + headline_body_first3sent."
        )
    )
    parser.add_argument("--train_csv", type=str, default="train_processed.csv")
    parser.add_argument("--val_csv", type=str, default="val_processed.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/fnc1_baseline")
    parser.add_argument(
        "--backend_name",
        type=str,
        default="tfidf_linear_svm",
        choices=TRAINABLE_BACKEND_CHOICES,
        help="Current trainable backends. This script intentionally keeps TF-IDF baselines only.",
    )
    parser.add_argument(
        "--text_mode",
        type=str,
        default="headline_body_first3sent",
        choices=TEXT_MODE_CHOICES,
        help="Text construction strategy for claim/evidence pair.",
    )
    parser.add_argument("--text_separator", type=str, default=" [SEP] ")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_features", type=int, default=120000)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=2)

    args = parser.parse_args()
    return StanceTrainingConfig(**vars(args))


def main() -> None:
    config = parse_args()
    results = run_training_pipeline(config)

    print("===== FNC-1 Modular Training Finished =====")
    print(f"backend: {config.backend_name}")
    print(f"text_mode: {config.text_mode}")
    print(f"4-way accuracy: {results['four_way']['accuracy']:.4f}")
    print(f"4-way macro_f1: {results['four_way']['macro_f1']:.4f}")

    for scheme_name, scheme_result in results["three_way"].items():
        print(
            f"{scheme_name} -> accuracy: {scheme_result['accuracy']:.4f}, "
            f"macro_f1: {scheme_result['macro_f1']:.4f}, "
            f"samples: {scheme_result['num_samples']}, dropped: {scheme_result['dropped_rows']}"
        )

    # Reusable module-level inference API.
    demo_result = predict_stance(
        claim="Climate change is accelerating.",
        evidence_text="Recent studies show global average temperatures continue to rise.",
        output_dir=config.output_dir,
    )
    print("\nInference demo:")
    print(demo_result)
    print("\nSaved artifacts:")
    print(f"output_dir: {results['artifacts']['output_dir']}")
    print(f"model_path: {results['artifacts']['model_path']}")
    print(f"prediction_csv: {results['artifacts']['prediction_csv']}")


if __name__ == "__main__":
    main()
