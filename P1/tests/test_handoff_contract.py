from __future__ import annotations

import unittest

from p1.handoff import pipeline_output_to_p2_payload
from p1.schemas import Claim, ClaimPair, ClaimSource, NLIPairResult, NliLabel, PipelineOutput


class HandoffContractTest(unittest.TestCase):
    def test_pipeline_output_to_p2_payload_preserves_core_fields(self) -> None:
        claim_a = Claim(
            claim_id="a",
            text="Claim A",
            source=ClaimSource(doc_id="doc-1", chunk_id="chunk-1", metadata={"rank": 1}),
            entities=["Alice"],
            subject="Alice",
            relation="said",
            object="something",
        )
        claim_b = Claim(
            claim_id="b",
            text="Claim B",
            source=ClaimSource(doc_id="doc-2", chunk_id="chunk-2"),
            entities=["Bob"],
            subject="Bob",
            relation="denied",
            object="it",
        )
        output = PipelineOutput(
            claims=[claim_a, claim_b],
            candidate_pairs=[
                ClaimPair(
                    claim_a=claim_a,
                    claim_b=claim_b,
                    entity_overlap=[],
                    lexical_similarity=0.42,
                )
            ],
            nli_results=[
                NLIPairResult(
                    claim_a_id="a",
                    claim_b_id="b",
                    entailment_score=0.1,
                    contradiction_score=0.8,
                    neutral_score=0.1,
                    label=NliLabel.CONTRADICTION,
                    metadata={"model": "heuristic"},
                )
            ],
        )

        payload = pipeline_output_to_p2_payload(output, sample_id="sample-1")

        self.assertEqual(payload["sample_id"], "sample-1")
        self.assertEqual(len(payload["claims"]), 2)
        self.assertEqual(payload["claims"][0]["claim_id"], "a")
        self.assertEqual(payload["claims"][0]["source_metadata"]["rank"], 1)
        self.assertEqual(payload["candidate_pairs"][0]["lexical_similarity"], 0.42)
        self.assertEqual(payload["nli_results"][0]["label"], "contradiction")


if __name__ == "__main__":
    unittest.main()
