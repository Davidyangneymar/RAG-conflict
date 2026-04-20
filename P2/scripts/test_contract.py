"""
Smoke-test the P1 -> P2 input contract.

Usage:
    python scripts/test_contract.py <path/to/p1_payload.json>

For each sample in the payload it prints:
    - sample_id
    - number of claims / candidate_pairs / nli_results
    - any nli_results whose claim ids do not appear in that sample's claims

Exit code:
    0 if every nli_result in every sample resolves to a known claim,
    1 otherwise (or on parse / IO error).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Let the script run without installing the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.p2.p1_adapter import load_p1_payload, P1SchemaError  # noqa: E402


def check_record(record) -> int:
    """Return the number of dangling claim-id references in this record."""
    print(f"sample_id: {record.sample_id}")
    print(f"  claims:          {len(record.claims)}")
    print(f"  candidate_pairs: {len(record.candidate_pairs)}")
    print(f"  nli_results:     {len(record.nli_results)}")

    dangling = 0
    for i, nli in enumerate(record.nli_results):
        missing = []
        if not record.has_claim(nli.claim_a_id):
            missing.append(("claim_a_id", nli.claim_a_id))
        if not record.has_claim(nli.claim_b_id):
            missing.append(("claim_b_id", nli.claim_b_id))
        if missing:
            dangling += 1
            parts = ", ".join(f"{k}={v!r}" for k, v in missing)
            print(
                f"  [ERROR] nli_results[{i}] references unknown claim id(s): {parts} "
                f"(label={nli.label!r})"
            )

    if dangling == 0:
        print("  [OK] all nli_results reference known claims.")
    else:
        print(f"  [FAIL] {dangling} nli_result(s) reference unknown claims.")
    return dangling


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    path = argv[1]
    try:
        records = load_p1_payload(path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1
    except P1SchemaError as e:
        print(f"[ERROR] P1 payload is malformed: {e}")
        return 1
    except ValueError as e:
        # json.JSONDecodeError inherits from ValueError
        print(f"[ERROR] could not parse JSON: {e}")
        return 1

    print(f"Loaded {len(records)} sample(s) from {path}\n")

    total_dangling = 0
    for record in records:
        total_dangling += check_record(record)
        print()

    if total_dangling == 0:
        print("All samples OK.")
        return 0
    print(f"Found {total_dangling} dangling nli_result reference(s) in total.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
