from __future__ import annotations

import unittest

import p1


class PublicAPITest(unittest.TestCase):
    def test_expected_symbols_are_exported(self) -> None:
        expected = {
            "Claim",
            "ClaimPair",
            "ClaimSource",
            "HeuristicNLIModel",
            "LLMClaimExtractor",
            "MultiStageBlocker",
            "NLIPairResult",
            "P1Pipeline",
            "SentenceClaimExtractor",
            "StructuredClaimExtractor",
            "RetrievedChunk",
            "RetrievalInput",
            "pipeline_output_to_p2_payload",
            "build_claim_extractor",
            "build_pipeline",
        }

        self.assertTrue(expected.issubset(set(p1.__all__)))
        for name in expected:
            self.assertTrue(hasattr(p1, name))

    def test_build_pipeline_returns_pipeline_instance(self) -> None:
        pipeline = p1.build_pipeline(extractor_kind="sentence")

        self.assertIsInstance(pipeline, p1.P1Pipeline)


if __name__ == "__main__":
    unittest.main()
