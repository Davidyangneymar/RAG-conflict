"""
End-to-end P2 closed-loop runner on AVeriTeC data.

AVeriTeC records  --(dev-only adapter)-->  P1-shape payload
                 --(p1_adapter)-->          InputRecord
                 --(stance runner)-->       StancedSample
                 --(conflict typer)-->      TypedSample

Writes:
  - out/p1_payload.json          the synthesized P1-shape payload
                                 (so you can diff against real P1 output later)
  - out/p2_final.json            the ConflictTypedOutput, ready for the
                                 conflict-strategy / generation module

Usage:
    python scripts/run_averitec_pipeline.py <averitec.json> \\
        [--out_dir outputs/averitec_demo] \\
        [--model_dir outputs/fnc1_bert_upgrade_full] [--device cpu|cuda|mps]

The input file is the standard AVeriTeC JSON (list of records or
{"data": [...]}).  You can try:
    python scripts/run_averitec_pipeline.py scripts/sample_averitec_records.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.p2 import (  # noqa: E402
    parse_p1_payload,
    run_full_p2_pipeline_from_records,
)
from src.p2.datasets import (  # noqa: E402
    averitec_records_to_p1_payload,
    load_averitec_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("averitec_path", help="Path to AVeriTeC JSON (list of records)")
    p.add_argument("--out_dir", default="outputs/averitec_demo")
    p.add_argument("--model_dir", default="")
    p.add_argument("--device", default="")
    p.add_argument("--max_records", type=int, default=0,
                   help="If >0, only process the first N records (for quick smoke).")
    return p.parse_args()


def print_typed_sample_summary(ts) -> None:
    print(f"sample_id: {ts.sample_id}  gold={ts.gold_verdict!r}")
    if ts.type_counts:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(ts.type_counts.items()))
        print(f"  type_counts: {summary}")
    for tp in ts.pair_results:
        sp = tp.stance
        print(
            f"    ({sp.claim_a_id} vs {sp.claim_b_id})  "
            f"stance={sp.stance_label}  nli={sp.nli_label}  "
            f"signal={sp.agreement_signal}  "
            f"-> type={tp.conflict_type} "
            f"(policy={tp.resolution_policy}, conf={tp.typing_confidence:.2f})"
        )
        for line in tp.rationale:
            print(f"        · {line}")


def verdict_vs_typecount_score(samples) -> dict:
    """
    Lightweight sanity scorer: does our per-sample type distribution
    roughly align with AVeriTeC's gold verdict?

    We count it as aligned if:
      Supported           -> any pair is 'none' or 'temporal_conflict'
      Refuted             -> any pair is 'hard_contradiction' or 'misinformation'
      Conflicting ...     -> any pair is 'ambiguity' or 'opinion_conflict'
                             or 'hard_contradiction'
      Not Enough Evidence -> all pairs 'noise' or no pairs

    This is NOT a benchmark metric — AVeriTeC is about verdict prediction
    while we are producing conflict types. It is a sanity check for the
    closed loop, nothing more.
    """
    aligned = {"Supported": 0, "Refuted": 0,
               "Conflicting Evidence/Cherrypicking": 0,
               "Not Enough Evidence": 0}
    total = dict(aligned)
    for ts in samples:
        gold = ts.gold_verdict
        if gold not in total:
            continue
        total[gold] += 1
        types = set(ts.type_counts.keys())
        ok = False
        if gold == "Supported":
            ok = bool(types & {"none", "temporal_conflict"})
        elif gold == "Refuted":
            ok = bool(types & {"hard_contradiction", "misinformation"})
        elif gold == "Conflicting Evidence/Cherrypicking":
            ok = bool(types & {"ambiguity", "opinion_conflict", "hard_contradiction"})
        elif gold == "Not Enough Evidence":
            ok = (not types) or (types <= {"noise"})
        if ok:
            aligned[gold] += 1
    return {
        "aligned": aligned,
        "total": total,
        "overall": sum(aligned.values()),
        "overall_total": sum(total.values()),
    }


def main() -> int:
    args = parse_args()

    raw = load_averitec_json(args.averitec_path)
    if args.max_records > 0:
        raw = raw[: args.max_records]
    p1_payload = averitec_records_to_p1_payload(raw)
    gold_verdicts = [rec.pop("_averitec_gold", None) for rec in p1_payload]

    # The adapter accepts a list of dicts and returns List[InputRecord].
    records = parse_p1_payload(p1_payload)

    typed = run_full_p2_pipeline_from_records(
        records,
        model_output_dir=args.model_dir or None,
        preferred_device=args.device or None,
        gold_verdicts=gold_verdicts,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Synthesized P1-shape payload (useful for diffing against P1 once it ships)
    (out_dir / "p1_payload.json").write_text(
        json.dumps(p1_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2) Final P2 output, the artifact P3 / conflict-strategy will consume
    (out_dir / "p2_final.json").write_text(
        json.dumps(typed.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Processed {len(typed.samples)} sample(s).\n")
    for ts in typed.samples:
        print_typed_sample_summary(ts)
        print()

    score = verdict_vs_typecount_score(typed.samples)
    print("=== Sanity check: type distribution vs AVeriTeC gold verdict ===")
    for k in score["total"]:
        if score["total"][k]:
            print(f"  {k:42s}  aligned {score['aligned'][k]}/{score['total'][k]}")
    print(f"  overall aligned: {score['overall']}/{score['overall_total']}")

    print(f"\nArtifacts written to {out_dir.resolve()}:")
    print(f"  - p1_payload.json   (synthesized P1 input)")
    print(f"  - p2_final.json     (ConflictTypedOutput)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
