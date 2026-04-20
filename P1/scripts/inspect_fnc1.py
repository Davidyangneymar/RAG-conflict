from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect normalized FNC-1 JSONL samples.")
    parser.add_argument(
        "--input",
        default="data/processed/fnc1_train.jsonl",
        help="Path to normalized FNC-1 JSONL file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of sample rows to print",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    label_counter: Counter[str] = Counter()
    stance_counter: Counter[str] = Counter()
    samples: list[dict] = []
    total = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            total += 1
            label_counter.update([record["nli_label"]])
            stance_counter.update([record["stance_label"]])
            if len(samples) < args.limit:
                samples.append(record)

    print(f"Input: {input_path}")
    print(f"Total records: {total}")
    print("\nNLI label distribution:")
    for label, count in sorted(label_counter.items()):
        print(f"  {label}: {count}")

    print("\nOriginal stance distribution:")
    for label, count in sorted(stance_counter.items()):
        print(f"  {label}: {count}")

    print("\nSample rows:")
    for index, sample in enumerate(samples, start=1):
        print(f"\n--- Sample {index} ---")
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
