"""
Analyze P2 pair-level CSV from inspect_p2_on_averitec.py.

Usage:
    python scripts/analyze_p2_pairs_csv.py outputs/p2_inspection_after_opt_full/pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_path", help="pairs.csv path")
    p.add_argument("--out", default="", help="optional output json path")
    return p.parse_args()


def _top_tokens(rows: List[Dict[str, str]], n: int = 80) -> List[List[object]]:
    c = Counter()
    stop = {
        "agreement", "signal", "neutral", "conflict", "stance", "type",
        "claim", "claims", "source", "quality", "detected", "looking",
        "falling", "further", "matched", "markers", "reported", "said",
        "news", "report", "reports", "according", "evidence", "none",
    }
    for r in rows:
        txt = (r.get("rationale") or "") + " || " + (r.get("agreement_signal") or "")
        for t in TOKEN_RE.findall(txt.lower()):
            if len(t) < 4 or t in stop:
                continue
            c[t] += 1
    return [[k, v] for k, v in c.most_common(n)]


def main() -> int:
    args = parse_args()
    path = Path(args.csv_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    by_gold_and_type: Dict[str, Counter] = defaultdict(Counter)
    by_rule: Counter = Counter()
    by_signal: Counter = Counter()
    refuted_none_rules: Counter = Counter()
    conflicting_none_rules: Counter = Counter()

    for r in rows:
        gold = (r.get("gold_verdict") or "").strip() or "(none)"
        ctype = (r.get("conflict_type") or "").strip() or "(none)"
        rule = (r.get("rule_fired") or "").strip() or "(none)"
        signal = (r.get("agreement_signal") or "").strip() or "(none)"

        by_gold_and_type[gold][ctype] += 1
        by_rule[rule] += 1
        by_signal[signal] += 1

        if gold == "Refuted" and ctype == "none":
            refuted_none_rules[rule] += 1
        if gold == "Conflicting Evidence/Cherrypicking" and ctype == "none":
            conflicting_none_rules[rule] += 1

    refuted_none_rows = [
        r for r in rows if (r.get("gold_verdict") == "Refuted" and r.get("conflict_type") == "none")
    ]
    conflicting_none_rows = [
        r for r in rows
        if (r.get("gold_verdict") == "Conflicting Evidence/Cherrypicking" and r.get("conflict_type") == "none")
    ]

    summary = {
        "rows": len(rows),
        "by_rule": by_rule,
        "by_signal": by_signal,
        "by_gold_and_type": {k: dict(v) for k, v in by_gold_and_type.items()},
        "refuted_none": {
            "count": len(refuted_none_rows),
            "rule_counter": dict(refuted_none_rules),
            "top_tokens": _top_tokens(refuted_none_rows),
        },
        "conflicting_none": {
            "count": len(conflicting_none_rows),
            "rule_counter": dict(conflicting_none_rules),
            "top_tokens": _top_tokens(conflicting_none_rows),
        },
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: dict(x))
    print(text)

    out = args.out.strip()
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"\nWrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
