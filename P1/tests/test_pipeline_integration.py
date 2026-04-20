from __future__ import annotations

import unittest

from p1.data.retrieval import normalize_retrieval_input
from p1.pipeline import build_pipeline


class PipelineIntegrationTest(unittest.TestCase):
    def test_sentence_pipeline_runs_retrieval_input_end_to_end(self) -> None:
        retrieval_input = normalize_retrieval_input(
            {
                "sample_id": "sample",
                "query": "Alice won the election.",
                "retrieved_chunks": [
                    {"chunk_id": "c1", "text": "Alice did not win the election.", "rank": 1}
                ],
            }
        )
        pipeline = build_pipeline(extractor_kind="sentence")

        output = pipeline.run_retrieval_input(retrieval_input)

        self.assertGreaterEqual(len(output.claims), 2)
        self.assertGreaterEqual(len(output.candidate_pairs), 1)
        self.assertEqual(len(output.candidate_pairs), len(output.nli_results))


if __name__ == "__main__":
    unittest.main()
