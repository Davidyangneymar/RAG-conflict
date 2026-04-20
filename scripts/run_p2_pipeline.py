"""
Run the P2 pipeline end-to-end: P1 JSON -> stance + NLI fusion -> P2 JSON.

Usage:
    python scripts/run_p2_pipeline.py <p1_payload.json> [--out p2_output.json] \\
        [--model_dir outputs/fnc1_bert_upgrade_full] [--device cpu|cuda|mps]

Prints a per-sample summary and, if --out is given, writes the full P2Output
as JSON so the conflict-typing module can consume it directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.p2 import run_p2_pipeline_from_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("payload", help="Path to P1 output JSON")
    p.add_argument("--out", default="", help="Optional output JSON path for P2Output")
    p.add_argument(
        "--model_dir",
        default="",
        help="BERT artifacts dir (defaults to fnc1_bert_stance_module.DEFAULT_BERT_OUTPUT_DIR)",
    )
    p.add_argument("--device", default="", help="Preferred device: cpu/cuda/mps")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    output = run_p2_pipeline_from_path(
        payload_path=args.payload,
        model_output_dir=args.model_dir or None,
        preferred_device=args.device or None,
    )

    for s in output.samples:
        print(f"sample_id: {s.sample_id}")
        print(
            f"  pairs: {s.num_pairs}  "
            f"agreement: {s.num_agreements}  "
            f"conflict: {s.num_conflicts}  "
            f"neutral: {s.num_neutral}  "
            f"unrelated: {s.num_unrelated}  "
            f"inconclusive: {s.num_inconclusive}"
        )
        for pr in s.pair_results:
            print(
                f"    ({pr.claim_a_id} vs {pr.claim_b_id})  "
                f"stance={pr.stance_label}  nli={pr.nli_label}  "
                f"-> {pr.agreement_signal} (conf={pr.fusion_confidence:.2f}, "
                f"dir={pr.stance_direction})"
            )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(output.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote P2Output -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
