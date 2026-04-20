from __future__ import annotations

import unittest

from p1.schemas import Claim, ClaimSource, NLIPairResult, NliLabel


class SchemasContractTest(unittest.TestCase):
    def test_claim_fills_flat_source_ids_from_source(self) -> None:
        claim = Claim(
            claim_id="c1",
            text="Alice won.",
            source=ClaimSource(doc_id="doc-1", chunk_id="chunk-1"),
        )

        self.assertEqual(claim.source_doc_id, "doc-1")
        self.assertEqual(claim.source_chunk_id, "chunk-1")

    def test_model_dump_preserves_nested_contract_fields(self) -> None:
        claim = Claim(
            claim_id="c2",
            text="Bob won.",
            source=ClaimSource(doc_id="doc-2", chunk_id="chunk-2", metadata={"role": "query"}),
        )

        payload = claim.model_dump()

        self.assertEqual(payload["claim_id"], "c2")
        self.assertEqual(payload["source"]["doc_id"], "doc-2")
        self.assertEqual(payload["source"]["metadata"]["role"], "query")

    def test_nli_label_is_still_available_as_value_in_contract(self) -> None:
        result = NLIPairResult(
            claim_a_id="a",
            claim_b_id="b",
            entailment_score=0.1,
            contradiction_score=0.8,
            neutral_score=0.1,
            label=NliLabel.CONTRADICTION,
        )

        self.assertEqual(result.label.value, "contradiction")


if __name__ == "__main__":
    unittest.main()
