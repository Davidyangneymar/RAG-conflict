from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from p5.adapters import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small annotation template from normalized benchmark data.")
    parser.add_argument("--input", required=True, help="Normalized benchmark JSONL path.")
    parser.add_argument("--output", default="data/annotation/small_sample_annotations.csv", help="Annotation CSV output.")
    parser.add_argument("--sample-size", type=int, default=60, help="Total sample size.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        label = str(row.get("gold_label", "neutral"))
        by_label.setdefault(label, []).append(row)

    random.seed(args.seed)
    labels = sorted(by_label.keys())
    if not labels:
        raise SystemExit("No rows found in input JSONL.")

    per_label = max(1, args.sample_size // len(labels))
    selected: list[dict] = []
    for label in labels:
        pool = by_label[label]
        k = min(per_label, len(pool))
        selected.extend(random.sample(pool, k=k))

    selected = selected[: args.sample_size]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "dataset",
                "query",
                "gold_label",
                "human_label",
                "is_conflict",
                "notes",
            ],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "sample_id": row.get("sample_id", ""),
                    "dataset": row.get("dataset", ""),
                    "query": row.get("query", ""),
                    "gold_label": row.get("gold_label", ""),
                    "human_label": "",
                    "is_conflict": "",
                    "notes": "",
                }
            )

    print(f"annotation_rows={len(selected)} output={output_path.as_posix()}")


if __name__ == "__main__":
    main()
