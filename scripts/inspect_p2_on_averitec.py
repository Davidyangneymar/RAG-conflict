"""
Batch introspection tool for P2 on AVeriTeC data.

NOT a benchmark. Designed for human eyeballing — produces:
  1. conflict_type distribution
  2. per-rule fire counter (which rules are dead / over-triggered)
  3. gold_verdict × conflict_type cross-table
  4. top-N "suspicious" samples for manual inspection
  5. a flat CSV (one row per pair) for P5 / spreadsheet analysis

Usage:
    # Quick: use the 4-record fixture (shows format but tiny signal)
    python scripts/inspect_p2_on_averitec.py scripts/sample_averitec_records.json

    # Real: download AVeriTeC dev first, then run at scale
    #   https://fever.ai/dataset/averitec.html
    #   or HuggingFace:  huggingface.co/datasets/pminervini/averitec
    python scripts/inspect_p2_on_averitec.py path/to/averitec_dev.json \\
        --sample_size 100 --top_n 10

Output:
    outputs/p2_inspection/
        pairs.csv        one row per (sample, pair) — feed to P5
        report.txt       the console report, saved for the log
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.p2 import parse_p1_payload, run_full_p2_pipeline_from_records  # noqa: E402
from src.p2.datasets import averitec_records_to_p1_payload, load_averitec_json  # noqa: E402


# ---------------------------------------------------------------------------
# Rule attribution — derive which rule fired from the rationale text.
# Keep this list ordered: first match wins (matches the order in typer.py).
# ---------------------------------------------------------------------------

RULE_MATCHERS: List[Tuple[str, str]] = [
    ("R0_missing_claim",        "could not be resolved"),
    ("R0a_signal_unrelated",    "stance filtered as unrelated"),
    ("R0e_source_gap_override", "strong source quality gap"),
    ("R0d_hidden_conflict_override", "agreement/neutral signal override"),
    ("R0b_signal_agreement",    "agreement_signal=agreement"),
    ("R0c_signal_neutral",      "agreement_signal=neutral"),
    ("R1_temporal",             "time markers differ"),
    ("R2_numerical",            "numerical clash"),
    ("R3_subject",              "subjects differ"),
    ("R4a_opinion_medium",      "op-ed / editorial medium"),
    ("R4b_opinion_text",        "opinion / reported-speech markers"),
    ("R5_misinfo",              "source quality gap"),
    ("R6_fallback",             "no further cues matched"),
]


def attribute_rule(rationale: List[str]) -> str:
    joined = " || ".join(rationale)
    for name, needle in RULE_MATCHERS:
        if needle in joined:
            return name
    return "R_unknown"


# ---------------------------------------------------------------------------
# Suspicious-sample heuristics (no F1 — just patterns worth eyeballing)
# ---------------------------------------------------------------------------

def is_suspicious(sample) -> Tuple[bool, str]:
    """Return (is_suspect, why) for a TypedSample vs its gold_verdict."""
    gold = sample.gold_verdict
    types = set(sample.type_counts.keys())
    if not types:
        return True, "no pairs at all"

    if gold == "Supported":
        bad = types & {"hard_contradiction", "misinformation"}
        if bad:
            return True, f"gold=Supported but pairs typed as {sorted(bad)}"
    if gold == "Refuted":
        if types <= {"none"}:
            return True, "gold=Refuted but every pair typed as 'none' (missed conflict)"
    if gold == "Conflicting Evidence/Cherrypicking":
        interesting = types & {
            "ambiguity", "opinion_conflict", "hard_contradiction", "temporal_conflict"
        }
        if not interesting:
            return True, "gold=Conflicting but no conflict-like type assigned"

    if types == {"hard_contradiction"} and len(sample.pair_results) > 0:
        # every pair fell through to the fallback — typer found no sub-signal
        all_fallback = all(
            attribute_rule(tp.rationale) == "R6_fallback"
            for tp in sample.pair_results
        )
        if all_fallback:
            return True, "all pairs hit R6_fallback (no sub-rule caught them)"

    return False, ""


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(typed_output, pair_rows: List[Dict[str, Any]], top_n: int) -> str:
    lines: List[str] = []
    lines.append("=" * 66)
    lines.append(" P2 batch inspection report")
    lines.append("=" * 66)

    lines.append(f"samples: {len(typed_output.samples)}  pairs: {len(pair_rows)}")
    lines.append("")

    # 1) conflict_type distribution
    ct_counter = Counter(r["conflict_type"] for r in pair_rows)
    lines.append("[1] conflict_type distribution (pair level)")
    for ct, n in ct_counter.most_common():
        pct = 100.0 * n / len(pair_rows) if pair_rows else 0.0
        lines.append(f"    {ct:22s}  {n:5d}   ({pct:5.1f}%)")
    lines.append("")

    # 2) Rule fire counter — which rules are DEAD or DOMINANT
    rule_counter = Counter(r["rule_fired"] for r in pair_rows)
    lines.append("[2] rule fire counter  (DEAD = 0 fires; DOMINANT = >50%)")
    defined = [name for name, _ in RULE_MATCHERS]
    for rule in defined + sorted(k for k in rule_counter if k not in defined):
        n = rule_counter.get(rule, 0)
        tag = ""
        if n == 0:
            tag = "  <- DEAD (rule never fired)"
        elif pair_rows and n / len(pair_rows) > 0.5:
            tag = "  <- DOMINANT"
        lines.append(f"    {rule:26s}  {n:5d}{tag}")
    lines.append("")

    # 3) verdict × type cross-table
    golds = sorted({r["gold_verdict"] for r in pair_rows if r["gold_verdict"]})
    types = sorted({r["conflict_type"] for r in pair_rows})
    if golds:
        lines.append("[3] gold_verdict × conflict_type  (pair counts)")
        header = "    " + "gold \\ type".ljust(40) + "  " + "  ".join(t[:10].ljust(10) for t in types)
        lines.append(header)
        for g in golds:
            row = [r for r in pair_rows if r["gold_verdict"] == g]
            cells = [str(sum(1 for r in row if r["conflict_type"] == t)).ljust(10) for t in types]
            lines.append(f"    {g[:40].ljust(40)}  " + "  ".join(cells))
        lines.append("")

    # 4) suspicious samples
    suspects: List[Tuple[str, str]] = []
    for s in typed_output.samples:
        ok, why = is_suspicious(s)
        if ok:
            suspects.append((s.sample_id, why))
    lines.append(f"[4] suspicious samples (top {top_n} of {len(suspects)}):")
    for sid, why in suspects[:top_n]:
        lines.append(f"    {sid}: {why}")
    if not suspects:
        lines.append("    (none — every sample's type set aligns with its gold verdict heuristic)")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="AVeriTeC JSON (list or {'data': [...]})")
    p.add_argument("--sample_size", type=int, default=0, help="Cap records; 0 = all")
    p.add_argument("--top_n", type=int, default=10, help="Top suspicious samples to print")
    p.add_argument("--out_dir", default="outputs/p2_inspection")
    p.add_argument("--model_dir", default="")
    p.add_argument("--device", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    raw = load_averitec_json(args.path)
    if args.sample_size > 0:
        raw = raw[: args.sample_size]
    p1_payload = averitec_records_to_p1_payload(raw)
    gold_verdicts = [rec.pop("_averitec_gold", None) for rec in p1_payload]
    records = parse_p1_payload(p1_payload)

    typed = run_full_p2_pipeline_from_records(
        records,
        model_output_dir=args.model_dir or None,
        preferred_device=args.device or None,
        gold_verdicts=gold_verdicts,
    )

    # Flatten to one row per pair for CSV + counter work.
    pair_rows: List[Dict[str, Any]] = []
    for sample in typed.samples:
        for tp in sample.pair_results:
            sp = tp.stance
            pair_rows.append({
                "sample_id": sample.sample_id,
                "gold_verdict": sample.gold_verdict,
                "claim_a_id": sp.claim_a_id,
                "claim_b_id": sp.claim_b_id,
                "stance_label": sp.stance_label,
                "stance_score": sp.stance_decision_score,
                "nli_label": sp.nli_label,
                "agreement_signal": sp.agreement_signal,
                "conflict_type": tp.conflict_type,
                "resolution_policy": tp.resolution_policy,
                "typing_confidence": tp.typing_confidence,
                "rule_fired": attribute_rule(tp.rationale),
                "rationale": " || ".join(tp.rationale),
            })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV (for P5 / spreadsheet eyeballing).
    csv_path = out_dir / "pairs.csv"
    if pair_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows)

    # Build + print + save report.
    report = format_report(typed, pair_rows, args.top_n)
    print(report)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")

    print(f"artifacts: {out_dir.resolve()}")
    print(f"  - pairs.csv    {len(pair_rows)} rows")
    print(f"  - report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
