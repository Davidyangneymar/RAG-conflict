from __future__ import annotations

import argparse
from pathlib import Path

from p5.adapters import normalize_records, normalized_to_dict, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize benchmark datasets for P5 evaluation.")
    parser.add_argument("--source", choices=["faitheval", "ambigdocs", "conflicts", "p1_benchmark"], required=True)
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output normalized JSONL path.")
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Allow missing labels and emit gold_label='__unlabeled__' (not valid for stance evaluation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    records = normalize_records(rows, dataset=args.source, allow_unlabeled=args.allow_unlabeled)
    write_jsonl(args.output, normalized_to_dict(records))

    print(f"normalized={len(records)} output={Path(args.output).as_posix()}")


if __name__ == "__main__":
    main()
