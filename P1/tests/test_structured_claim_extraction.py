from __future__ import annotations

import unittest
from unittest.mock import patch

from p1.claim_extraction import LLMClaimExtractor, SentenceClaimExtractor, StructuredClaimExtractor
from p1.schemas import ChunkInput


class StructuredClaimExtractionTest(unittest.TestCase):
    def test_structured_extractor_fills_subject_relation_object(self) -> None:
        extractor = StructuredClaimExtractor(entity_backend="regex")

        claims = extractor.extract(
            ChunkInput(
                doc_id="doc",
                chunk_id="chunk",
                text="Alice said Bob won in 2024.",
            )
        )

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].subject, "Alice")
        self.assertEqual(claims[0].relation, "said")
        self.assertIn("Bob", claims[0].object or "")
        self.assertTrue(claims[0].metadata["structured_fields_present"])

    def test_sentence_extractor_infers_negative_polarity(self) -> None:
        claim = SentenceClaimExtractor(entity_backend="regex").extract(
            ChunkInput(doc_id="doc", chunk_id="chunk", text="Alice did not approve the plan.")
        )[0]

        self.assertEqual(claim.polarity, "negative")
        self.assertEqual(claim.certainty, 0.7)

    def test_sentence_extractor_infers_uncertainty(self) -> None:
        claim = SentenceClaimExtractor(entity_backend="regex").extract(
            ChunkInput(doc_id="doc", chunk_id="chunk", text="Alice may approve the plan.")
        )[0]

        self.assertEqual(claim.polarity, "uncertain")
        self.assertEqual(claim.certainty, 0.45)

    def test_sentence_split_keeps_common_abbreviation(self) -> None:
        claims = SentenceClaimExtractor(entity_backend="regex").extract(
            ChunkInput(doc_id="doc", chunk_id="chunk", text="Dr. Smith was present. Alice was absent.")
        )

        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0].text, "Dr. Smith was present.")

    def test_llm_extractor_logs_missing_runtime_without_network(self) -> None:
        with patch.dict(
            "os.environ",
            {"P1_LLM_API_KEY": "", "OPENAI_API_KEY": "", "P1_LLM_MODEL": ""},
        ):
            extractor = LLMClaimExtractor(entity_backend="regex", fallback_to_heuristic=True)
            with self.assertLogs("p1.claim_extraction", level="INFO") as captured:
                claims = extractor.extract(
                    ChunkInput(doc_id="doc", chunk_id="chunk", text="Alice said Bob won.")
                )

        self.assertEqual(len(claims), 1)
        self.assertTrue(any("event=p1.claim_extraction.llm.complete" in message for message in captured.output))
        self.assertTrue(any("duration_ms=" in message for message in captured.output))


if __name__ == "__main__":
    unittest.main()
