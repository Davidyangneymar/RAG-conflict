from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.data.fnc1 import convert_fnc1, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize FNC-1 raw CSV files into JSONL.")
    parser.add_argument("--bodies", required=True, help="Path to train_bodies.csv or competition_test_bodies.csv")
    parser.add_argument("--stances", required=True, help="Path to train_stances.csv or competition_test_stances.csv")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = convert_fnc1(args.bodies, args.stances)
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} normalized records to {args.output}")


if __name__ == "__main__":
    main()
