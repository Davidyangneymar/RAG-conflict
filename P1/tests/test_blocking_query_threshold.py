from __future__ import annotations

import unittest

from p1.blocking import BlockingConfig, MultiStageBlocker
from p1.schemas import Claim, ClaimSource


def make_claim(claim_id: str, text: str, role: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        source=ClaimSource(doc_id=claim_id, metadata={"role": role}),
    )


class BlockingQueryThresholdTest(unittest.TestCase):
    def test_query_pair_threshold_can_relax_default_lexical_gate(self) -> None:
        query = make_claim("q", "What sport is Bobby Carpenter associated with?", "query")
        evidence = make_claim("e", "Bobby Carpenter played football professionally.", "retrieved_evidence")

        strict_blocker = MultiStageBlocker(config=BlockingConfig(min_lexical_similarity=0.3))
        relaxed_blocker = MultiStageBlocker(
            config=BlockingConfig(
                min_lexical_similarity=0.3,
                query_pair_min_lexical_similarity=0.05,
            )
        )

        self.assertIsNone(strict_blocker._build_pair(query, evidence))
        self.assertIsNotNone(relaxed_blocker._build_pair(query, evidence))

    def test_query_pair_threshold_does_not_apply_to_evidence_evidence_pairs(self) -> None:
        claim_a = make_claim("a", "Bobby Carpenter played football professionally.", "retrieved_evidence")
        claim_b = make_claim("b", "Carpenter was an athlete.", "retrieved_evidence")
        blocker = MultiStageBlocker(
            config=BlockingConfig(
                min_lexical_similarity=0.3,
                query_pair_min_lexical_similarity=0.05,
            )
        )

        self.assertIsNone(blocker._build_pair(claim_a, claim_b))


if __name__ == "__main__":
    unittest.main()
