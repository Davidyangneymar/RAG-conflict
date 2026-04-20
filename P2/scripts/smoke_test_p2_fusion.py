"""
Fast unit smoke test for the P2 stance/NLI fusion layer and role logic.

Does NOT load the BERT model — only exercises pure Python.

Run:
    python scripts/smoke_test_p2_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.p2 import (  # noqa: E402
    Claim,
    decide_claim_evidence_roles,
    fuse_stance_and_nli,
)


def test_role_query_vs_evidence() -> None:
    a = Claim(claim_id="q", text="Q", role="query")
    b = Claim(claim_id="e", text="E", role="retrieved_evidence")
    claim, evidence, direction = decide_claim_evidence_roles(a, b)
    assert claim.claim_id == "q"
    assert evidence.claim_id == "e"
    assert direction == "a_as_claim"

    claim, evidence, direction = decide_claim_evidence_roles(b, a)
    assert claim.claim_id == "q"
    assert evidence.claim_id == "e"
    assert direction == "b_as_claim"


def test_role_headline_vs_body() -> None:
    a = Claim(claim_id="h", text="H", role="headline")
    b = Claim(claim_id="body", text="B", role="body")
    claim, _, direction = decide_claim_evidence_roles(a, b)
    assert claim.claim_id == "h"
    assert direction == "a_as_claim"


def test_role_unknown_falls_back() -> None:
    a = Claim(claim_id="a", text="A")
    b = Claim(claim_id="b", text="B")
    _, _, direction = decide_claim_evidence_roles(a, b)
    assert direction == "bidirectional"


def test_fusion_both_agree() -> None:
    sig, conf, _ = fuse_stance_and_nli("support", 0.9, "entailment")
    assert sig == "agreement"
    assert conf > 0.9

    sig, _, _ = fuse_stance_and_nli("oppose", 0.8, "contradiction")
    assert sig == "conflict"

    sig, _, _ = fuse_stance_and_nli("neutral", 0.6, "neutral")
    assert sig == "neutral"


def test_fusion_filtered_short_circuits() -> None:
    sig, _, notes = fuse_stance_and_nli("filtered", 0.5, "contradiction")
    assert sig == "unrelated"
    assert any("filtered" in n for n in notes)


def test_fusion_disagreement_is_inconclusive() -> None:
    sig, _, notes = fuse_stance_and_nli("support", 0.9, "contradiction")
    assert sig == "inconclusive"
    assert any("disagrees" in n for n in notes)


def test_fusion_missing_nli_uses_stance() -> None:
    sig, _, notes = fuse_stance_and_nli("oppose", 0.7, None)
    assert sig == "conflict"
    assert any("no NLI" in n for n in notes)


def test_fusion_missing_stance_uses_nli() -> None:
    sig, _, notes = fuse_stance_and_nli(None, None, "entailment")
    assert sig == "agreement"
    assert any("no stance" in n for n in notes)


def main() -> int:
    tests = [
        test_role_query_vs_evidence,
        test_role_headline_vs_body,
        test_role_unknown_falls_back,
        test_fusion_both_agree,
        test_fusion_filtered_short_circuits,
        test_fusion_disagreement_is_inconclusive,
        test_fusion_missing_nli_uses_stance,
        test_fusion_missing_stance_uses_nli,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL {t.__name__}: {e}")
    if failures:
        print(f"\n{failures} failure(s).")
        return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
