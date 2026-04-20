"""
Profile source metadata in AVeriTeC dev records.

Usage:
    python scripts/profile_averitec_sources.py scripts/dev.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="Path to AVeriTeC JSON list")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(Path(args.path).read_text(encoding="utf-8"))

    mediums = Counter()
    domains = Counter()

    for rec in data:
        for q in rec.get("questions") or []:
            for ans in q.get("answers") or []:
                m = str(ans.get("source_medium") or "").strip().lower()
                if m:
                    mediums[m] += 1

                u = str(ans.get("source_url") or "").strip()
                if u:
                    host = urlparse(u).netloc.lower()
                    if host.startswith("www."):
                        host = host[4:]
                    if host:
                        domains[host] += 1

    print("top source_medium values:")
    for k, v in mediums.most_common(30):
        print(f"  {k}\t{v}")

    print("\ntop source domains:")
    for k, v in domains.most_common(40):
        print(f"  {k}\t{v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
