from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P1 export_p5_benchmark.py using only relative paths.")
    parser.add_argument("--python", required=True, help="Python executable path (e.g. P5/.venv_p123/Scripts/python.exe).")
    parser.add_argument("--dataset", choices=["fnc1", "averitec", "retrieval_json"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    p5_root = Path(__file__).parent.parent
    script = Path("..") / "P1" / "scripts" / "export_p5_benchmark.py"
    cmd = [
        args.python,
        str(script),
        "--dataset",
        args.dataset,
        "--input",
        args.input,
        "--output",
        args.output,
        "--limit",
        str(args.limit),
    ]
    subprocess.run(cmd, check=True, cwd=p5_root)
    print(f"p1_export_done output={Path(args.output).as_posix()}")


if __name__ == "__main__":
    main()
