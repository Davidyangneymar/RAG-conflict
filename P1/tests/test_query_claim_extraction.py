from __future__ import annotations

import unittest

from p1.claim_extraction import SentenceClaimExtractor
from p1.schemas import ChunkInput


class QueryClaimExtractionTest(unittest.TestCase):
    def test_query_role_keeps_question_like_text(self) -> None:
        extractor = SentenceClaimExtractor()
        claims = extractor.extract(
            ChunkInput(
                doc_id="sample",
                chunk_id="query",
                text="What sport is Bobby Carpenter associated with?",
                metadata={"role": "query"},
            )
        )
        self.assertEqual(len(claims), 1)
        self.assertIn("What sport", claims[0].text)

    def test_non_query_question_like_text_is_still_filtered(self) -> None:
        extractor = SentenceClaimExtractor()
        claims = extractor.extract(
            ChunkInput(
                doc_id="sample",
                chunk_id="evidence",
                text="What sport is Bobby Carpenter associated with?",
                metadata={"role": "retrieved_evidence"},
            )
        )
        self.assertEqual(claims, [])


if __name__ == "__main__":
    unittest.main()
