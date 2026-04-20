"""Run bootstrap significance tests on a saved evidence-selection ablation.

Reads the JSON written by `eval_evidence_selection.py --output-json` and
reports:
    - 95% CI on macro-F1 for every body mode (single-system bootstrap)
    - Paired bootstrap test of every CCES mode against the top2_span baseline

The JSON must include per-sample predictions (added by the latest version
of the eval script). Older JSON files without `predictions` are skipped
with a warning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1.stats import bootstrap_macro_f1_ci, paired_bootstrap_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap significance for CCES ablation results.")
    parser.add_argument("--input", required=True, help="Path to the eval JSON")
    parser.add_argument("--baseline-mode", default="top2_span")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = payload["results"]

    by_mode = {row["body_mode"]: row for row in results}
    if "predictions" not in by_mode.get(args.baseline_mode, {}):
        print(
            f"ERROR: baseline mode '{args.baseline_mode}' has no 'predictions' field. "
            "Re-run eval_evidence_selection.py to regenerate the JSON with per-sample predictions."
        )
        raise SystemExit(2)

    baseline_preds = [p["pred"] for p in by_mode[args.baseline_mode]["predictions"]]
    gold = [p["gold"] for p in by_mode[args.baseline_mode]["predictions"]]
    n = len(gold)
    print(f"Loaded {n} samples from {args.input}")
    print(f"Baseline: {args.baseline_mode}  Iterations: {args.iterations}\n")

    print("=" * 70)
    print(f"{'mode':<22} {'macro_f1':>10} {'95% CI':>22}")
    print("-" * 70)
    per_mode_ci: dict[str, dict] = {}
    for row in results:
        if "predictions" not in row:
            continue
        preds = [p["pred"] for p in row["predictions"]]
        gold_for_mode = [p["gold"] for p in row["predictions"]]
        ci = bootstrap_macro_f1_ci(
            gold_for_mode, preds, iterations=args.iterations, seed=args.seed
        )
        per_mode_ci[row["body_mode"]] = ci
        print(
            f"{row['body_mode']:<22} {ci['point']:>10.4f}   "
            f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
        )

    print("\n" + "=" * 70)
    print(f"Paired bootstrap vs {args.baseline_mode}")
    print("=" * 70)
    print(f"{'mode':<22} {'Δ macro_f1':>12} {'95% CI':>26} {'p (≤0)':>10}")
    print("-" * 70)
    paired_results: dict[str, dict] = {}
    for row in results:
        if row["body_mode"] == args.baseline_mode:
            continue
        if "predictions" not in row:
            continue
        candidate_preds = [p["pred"] for p in row["predictions"]]
        # Sanity: ids must align across modes
        candidate_ids = [p["sample_id"] for p in row["predictions"]]
        baseline_ids = [p["sample_id"] for p in by_mode[args.baseline_mode]["predictions"]]
        if candidate_ids != baseline_ids:
            print(f"  WARN: sample_id mismatch for {row['body_mode']}, skipping")
            continue
        result = paired_bootstrap_test(
            gold,
            baseline_preds,
            candidate_preds,
            iterations=args.iterations,
            seed=args.seed,
        )
        paired_results[row["body_mode"]] = result
        marker = "***" if result["p_value"] < 0.001 else ("**" if result["p_value"] < 0.01 else ("*" if result["p_value"] < 0.05 else " "))
        print(
            f"{row['body_mode']:<22} {result['delta']:>+12.4f}   "
            f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]   "
            f"{result['p_value']:>8.4f} {marker}"
        )

    print("\nLegend: *** p<0.001, ** p<0.01, * p<0.05")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(
                {
                    "input": args.input,
                    "baseline_mode": args.baseline_mode,
                    "iterations": args.iterations,
                    "seed": args.seed,
                    "n": n,
                    "per_mode_ci": per_mode_ci,
                    "paired_vs_baseline": paired_results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote bootstrap report to {args.output_json}")


if __name__ == "__main__":
    main()
